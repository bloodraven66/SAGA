"use client";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { useCallback, useEffect, useRef, useState } from "react";
import { useMicrophoneAccess } from "./useMicrophoneAccess";
import { base64DecodeOpus, base64EncodeOpus } from "./audioUtil";
import { useAudioProcessor } from "./useAudioProcessor";
import CouldNotConnect, { HealthStatus } from "./CouldNotConnect";
import useWakeLock from "./useWakeLock";
import ErrorMessages, { ErrorItem, makeErrorItem } from "./ErrorMessages";
import { useBackendServerUrl } from "./useBackendServerUrl";
import { RECORDING_CONSENT_STORAGE_KEY } from "./ConsentModal";
import { getCSSVariable } from "./cssUtil";

// Offline-safe: NO next/font/google (that import crashes the dev server on a
// node with no internet). System font stacks only.
const SAGA_VOICE = "unmute-prod-website/p329_022.wav";

type Paper = { title: string; authors: string; year: number | null; citations: string | number | null };
type ToolCall = {
  id: string | null;
  tool: string;
  query: string;
  freshness: number | null;
  impact: number | null;
  count?: number;
  error?: string | null;
  papers?: Paper[];
  status: "running" | "done";
};
// A group = one decompose turn: its spoken note + the parallel tool calls it
// spawned. Tagged with `conv` (which conversation it belongs to) so the track
// can persist groups ACROSS turns and dim them by age instead of wiping.
type Grp = { key: string; conv: number; turn: number; note: string; calls: ToolCall[] };
type Phase = "idle" | "listening" | "thinking" | "searching" | "answering" | "done" | "interrupted";
// The two orbs + transcript bubbles are the persistent stage; only `groups`
// carries history across turns.
type Stage = {
  conv: number; user: string; query: string; answer: string; phase: Phase; groups: Grp[];
  // Speculation: `speculating` is the live pre-VAD anticipation (system started
  // synthesizing a filler before the user finished). `specNote` is the committed
  // anticipated opener that actually played; `specHyp` is the partial ASR
  // hypothesis it was predicted from.
  speculating: boolean; specNote: string; specHyp: string;
};
const emptyStage = (): Stage => ({ conv: 0, user: "", query: "", answer: "", phase: "idle", groups: [], speculating: false, specNote: "", specHyp: "" });
const MID_TURN: Phase[] = ["thinking", "searching", "answering"];
const MAX_GROUPS = 12;

const citedInAnswer = (paper: Paper, answer: string): boolean => {
  if (!answer) return false;
  const a = answer.toLowerCase();
  return paper.title.toLowerCase().split(/[^a-z0-9]+/).some((w) => w.length > 5 && a.includes(w));
};

// Dim by rank among the turns actually present (newest = full), so the fade is
// robust to gaps in the conv counter. The current turn (age 0) stays fully vivid;
// every PAST turn drops hard so it recedes into background context and the live
// tool calls clearly own the foreground (also stops the interrupted-mid-turn view
// from looking cluttered). Pair with the `.past` grayscale treatment in CSS.
const ageOpacity = (age: number): number => {
  if (age <= 0) return 1;
  return Math.max(0.1, 0.32 - (age - 1) * 0.1);
};

// Isolated so its per-tick typewriter re-renders don't touch the orbs/track.
const AnswerBubble = ({ text }: { text: string }) => {
  const [n, setN] = useState(0);
  useEffect(() => {
    setN(0);
    if (!text) return;
    if (matchMedia("(prefers-reduced-motion:reduce)").matches) { setN(text.length); return; }
    let i = 0;
    const id = setInterval(() => { i += 2; setN(i); if (i >= text.length) clearInterval(id); }, 45);
    return () => clearInterval(id);
  }, [text]);
  const shown = text.slice(0, n);
  return (
    <div className="saga-bubble agent">
      <span className="who">SAGA · grounded answer</span>
      <p className="txt">{shown}{shown.length < text.length && <span className="caret">▍</span>}</p>
    </div>
  );
};

// A persistent, audio-reactive orb. Reads the live AnalyserNode every frame
// (via the audio-processor ref) so it keeps moving regardless of React renders.
const hexToRgb = (hex: string): [number, number, number] => {
  const h = hex.replace("#", "").trim();
  const s = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const n = parseInt(s || "888888", 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

const SagaOrb = ({ proc, which, colorVar, active, connected }: {
  proc: ReturnType<typeof useAudioProcessor>["audioProcessor"];
  which: "input" | "output";
  colorVar: string;
  active: boolean;
  connected: boolean;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Live prop mirrors read inside the rAF loop (synced every render).
  const activeRef = useRef(active);
  const connRef = useRef(connected);
  activeRef.current = active;
  connRef.current = connected;

  return <OrbCanvas canvasRef={canvasRef} proc={proc} which={which} colorVar={colorVar} activeRef={activeRef} connRef={connRef} />;
};

// Split out so the parent can update the refs each render without restarting rAF.
const OrbCanvas = ({ canvasRef, proc, which, colorVar, activeRef, connRef }: {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  proc: ReturnType<typeof useAudioProcessor>["audioProcessor"];
  which: "input" | "output";
  colorVar: string;
  activeRef: React.MutableRefObject<boolean>;
  connRef: React.MutableRefObject<boolean>;
}) => {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const [r, g, b] = hexToRgb(getCSSVariable(colorVar) || "#c97e1e");
    const N = 180;
    const ring = new Float32Array(N);
    const buf = new Float32Array(2048);
    const freq = new Float32Array(1024);
    let breath = 0, spin = 0, size = 0, raf = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const parent = canvas.parentElement;
      size = parent ? Math.min(parent.clientWidth, parent.clientHeight) : 160;
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    if (canvas.parentElement) ro.observe(canvas.parentElement);

    const frame = () => {
      const a = which === "input" ? proc.current?.inputAnalyser ?? null : proc.current?.outputAnalyser ?? null;
      let energy = 0;
      if (a) {
        a.fftSize = 2048;
        a.smoothingTimeConstant = 0.82;
        a.getFloatTimeDomainData(buf);
        a.getFloatFrequencyData(freq);
        let s = 0;
        for (let i = 0; i < freq.length; i++) s += Math.max(-100, freq[i]);
        energy = Math.max(0, (s / freq.length + 100) / 65); // ~0..1.5
      }
      breath += 0.018;
      spin += 0.0016;
      const act = activeRef.current ? 1 : 0.32;
      const idle = 0.5 + 0.5 * Math.sin(breath);
      const cx = size / 2, cy = size / 2, base = size * 0.30;
      ctx.clearRect(0, 0, size, size);

      // outer glow
      const glowR = base * (1.55 + 0.5 * energy);
      const gg = ctx.createRadialGradient(cx, cy, base * 0.2, cx, cy, glowR);
      gg.addColorStop(0, `rgba(${r},${g},${b},${0.20 * act + 0.16 * energy})`);
      gg.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.fillStyle = gg;
      ctx.beginPath(); ctx.arc(cx, cy, glowR, 0, Math.PI * 2); ctx.fill();

      // reactive ring (waveform-modulated radius) + soft core fill
      ctx.beginPath();
      for (let i = 0; i <= N; i++) {
        const idx = i % N;
        const w = a ? buf[Math.floor((idx / N) * buf.length)] : 0;
        ring[idx] = ring[idx] * 0.6 + w * 0.4;
        const amp = 0.86 + 0.30 * Math.tanh(ring[idx] * 3 * (0.5 + act));
        const wob = 1 + 0.028 * Math.sin(idx * 3 + breath * 2) * idle * (1 - Math.min(1, energy));
        const rad = base * amp * wob * (1 + 0.06 * energy);
        const ang = (idx / N) * Math.PI * 2 + spin;
        const x = cx + rad * Math.cos(ang), y = cy + rad * Math.sin(ang);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, base * 1.2);
      cg.addColorStop(0, `rgba(${r},${g},${b},${0.28 + 0.26 * act})`);
      cg.addColorStop(0.7, `rgba(${r},${g},${b},${0.12 + 0.12 * act})`);
      cg.addColorStop(1, `rgba(${r},${g},${b},0.02)`);
      ctx.fillStyle = cg; ctx.fill();
      ctx.strokeStyle = `rgba(${r},${g},${b},${0.5 + 0.42 * act})`;
      ctx.lineWidth = 1.5 + 1.8 * act + 1.8 * energy;
      ctx.stroke();

      // bright inner nucleus
      const dotR = base * (0.28 + 0.08 * energy) * (connRef.current ? 1 : 0.7);
      const dg = ctx.createRadialGradient(cx, cy, 0, cx, cy, dotR);
      dg.addColorStop(0, `rgba(255,251,246,${0.85 * act + 0.12})`);
      dg.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.fillStyle = dg;
      ctx.beginPath(); ctx.arc(cx, cy, dotR, 0, Math.PI * 2); ctx.fill();

      raf = requestAnimationFrame(frame);
    };
    frame();
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <canvas ref={canvasRef} className="saga-orb-canvas" />;
};

const Saga = () => {
  const [stage, setStage] = useState<Stage>(emptyStage());

  const { microphoneAccess, askMicrophoneAccess } = useMicrophoneAccess();
  const [shouldConnect, setShouldConnect] = useState(false);
  const backendServerUrl = useBackendServerUrl();
  const [webSocketUrl, setWebSocketUrl] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [errors, setErrors] = useState<ErrorItem[]>([]);

  // Horizontal conveyor: translate the inner track so the newest group is
  // pinned right (older slide off left) or centered when it fits.
  const trackRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);

  useWakeLock(shouldConnect);

  useEffect(() => {
    if (!backendServerUrl) return;
    setWebSocketUrl(backendServerUrl.toString() + "/v1/realtime");
    (async () => {
      try {
        const controller = new AbortController();
        const t = setTimeout(() => controller.abort(), 3000);
        const r = await fetch(`${backendServerUrl}/v1/health`, { signal: controller.signal });
        clearTimeout(t);
        if (!r.ok) return setHealthStatus({ connected: "yes_request_fail", ok: false });
        const data = await r.json();
        data["connected"] = "yes_request_ok";
        setHealthStatus(data);
      } catch {
        setHealthStatus({ connected: "no", ok: false });
      }
    })();
  }, [backendServerUrl]);

  const handleAgentic = useCallback((kind: string, d: Record<string, unknown>) => {
    setStage((s) => {
      switch (kind) {
        case "speculating":
          // Pre-VAD: the system started synthesizing a filler off the partial ASR
          // hypothesis, before the user even finished. The wow moment.
          return { ...s, speculating: true, specNote: "", specHyp: (d.transcript as string) || "" };
        case "turn_start":
          // Authoritative new-turn signal: bump conv so THIS turn's groups get a
          // unique (conv,turn) key/tag -- independent of whether the barge-in
          // transcription fired. Keep groups (they dim with age).
          return { ...s, conv: s.conv + 1, query: (d.query as string) || s.user, answer: "", phase: "thinking" };
        case "tool_calls": {
          const turnIdx = d.turn as number;
          const calls = (d.calls as ToolCall[]).map((c) => ({ ...c, status: "running" as const }));
          const g: Grp = { key: `${s.conv}-${turnIdx}`, conv: s.conv, turn: turnIdx, note: "", calls };
          return { ...s, groups: [...s.groups, g].slice(-MAX_GROUPS), phase: "searching" };
        }
        case "tool_results": {
          const turnIdx = d.turn as number;
          const results = d.results as { id: string | null; tool: string; count: number; error: string | null; papers: Paper[] }[];
          const groups = s.groups.map((g) => {
            if (g.conv !== s.conv || g.turn !== turnIdx) return g;
            const used = new Set<number>();
            const calls = g.calls.map((c) => {
              let idx = results.findIndex((r, i) => r.id === c.id && !used.has(i));
              if (idx < 0) idx = results.findIndex((r, i) => r.tool === c.tool && !used.has(i));
              if (idx < 0) return c;
              used.add(idx);
              const r = results[idx];
              return { ...c, count: r.count, error: r.error, papers: r.papers, status: "done" as const };
            });
            return { ...g, calls };
          });
          return { ...s, groups };
        }
        case "note": {
          const turnIdx = d.turn as number;
          // turn 0 = the anticipated speculative opener; it has no tool group, so
          // surface it as the committed speculation next to the SAGA orb (the live
          // "anticipating…" resolves into "anticipated · <note>").
          if (turnIdx === 0 || d.speculative) {
            return { ...s, speculating: false, specNote: (d.text as string) || "", specHyp: (d.hypothesis as string) || s.specHyp };
          }
          return { ...s, groups: s.groups.map((g) => (g.conv === s.conv && g.turn === turnIdx ? { ...g, note: d.text as string } : g)) };
        }
        case "answer":
          return { ...s, answer: d.text as string, phase: "answering" };
        case "turn_end":
          return { ...s, phase: s.phase === "interrupted" ? s.phase : "done" };
        case "interrupted": {
          // A turn interrupted BEFORE it answered was a speculative/aborted
          // attempt (e.g. premature endpoint during a pause, or a correction) --
          // drop its cards rather than leaving a phantom dimmed "past turn".
          // Keep them if an answer was already produced (a real, if partial,
          // exchange).
          const groups = s.answer ? s.groups : s.groups.filter((g) => g.conv !== s.conv);
          return { ...s, groups, phase: "interrupted", speculating: false };
        }
        default:
          return s;
      }
    });
  }, []);

  // Single message router — fires for EVERY message (no lastMessage coalescing),
  // and decodes audio immediately without waiting on a React render.
  const audioRef = useRef<ReturnType<typeof useAudioProcessor>["audioProcessor"]>(null as never);
  const onMessage = useCallback((event: MessageEvent) => {
    let data: { type: string; [k: string]: unknown };
    try { data = JSON.parse(event.data); } catch { return; }
    if (data.type === "response.audio.delta") {
      const opus = base64DecodeOpus(data.delta as string);
      const ap = audioRef.current?.current;
      if (ap) ap.decoder.postMessage({ command: "decode", pages: opus }, [opus.buffer]);
    } else if (data.type === "unmute.agentic.update") {
      handleAgentic(data.kind as string, (data.data as Record<string, unknown>) || {});
    } else if (data.type === "unmute.interrupted_by_vad") {
      // Backend dumps buffered opener audio all-at-once (avoids the click caused
      // by two independently-paced streams merging at the seam), which means the
      // client can be holding several seconds of not-yet-played audio at barge-in
      // time. Flush the worklet's buffer immediately so interruption stays snappy
      // regardless of how deep that buffer got.
      audioRef.current?.current?.outputWorklet.port.postMessage({ type: "reset" });
    } else if (data.type === "conversation.item.input_audio_transcription.delta") {
      // STT deltas are whole words with no surrounding whitespace, so space-join
      // them (matches the reference frontend, which newline-joins per word).
      const delta = (data.delta as string).trim();
      if (!delta) return;
      setStage((s) => {
        if (MID_TURN.includes(s.phase)) return s; // ignore STT flush mid-turn
        const fresh = s.phase === "done" || s.phase === "interrupted";
        if (fresh) {
          // A new utterance begins: reset the bubbles + last turn's speculation
          // (the new utterance's `speculating` event fires shortly after, once
          // anticipation crosses threshold) but keep the persistent groups. conv
          // is bumped authoritatively at turn_start (not here), so a swallowed
          // barge-in transcript can't cause a (conv,turn) collision.
          return { ...s, user: delta, query: "", answer: "", phase: "listening", speculating: false, specNote: "", specHyp: "" };
        }
        return { ...s, user: s.user ? s.user + " " + delta : delta, phase: "listening" };
      });
    } else if (data.type === "error") {
      const err = data.error as { type: string; message: string };
      if (err.type !== "warning") setErrors((prev) => [...prev, makeErrorItem(err.message)]);
    }
  }, [handleAgentic]);

  const { sendMessage, readyState } = useWebSocket(
    webSocketUrl || null,
    { protocols: ["realtime"], onMessage },
    shouldConnect
  );

  const onOpusRecorded = useCallback(
    (opus: Uint8Array) => sendMessage(JSON.stringify({ type: "input_audio_buffer.append", audio: base64EncodeOpus(opus) })),
    [sendMessage]
  );
  const { setupAudio, shutdownAudio, audioProcessor } = useAudioProcessor(onOpusRecorded);
  audioRef.current = audioProcessor;

  const onConnectButtonPress = async () => {
    if (!shouldConnect) {
      const mediaStream = await askMicrophoneAccess();
      if (mediaStream) { await setupAudio(mediaStream); setShouldConnect(true); }
    } else {
      setShouldConnect(false);
      shutdownAudio();
    }
  };

  useEffect(() => {
    if (readyState === ReadyState.CLOSING || readyState === ReadyState.CLOSED) {
      setShouldConnect(false);
      shutdownAudio();
    }
  }, [readyState, shutdownAudio]);

  useEffect(() => {
    if (readyState !== ReadyState.OPEN) return;
    const consent = localStorage.getItem(RECORDING_CONSENT_STORAGE_KEY) === "true";
    setStage(emptyStage());
    sendMessage(JSON.stringify({ type: "session.update", session: { voice: SAGA_VOICE, allow_recording: consent } }));
  }, [readyState, sendMessage]);

  // Recompute the conveyor position whenever the number/shape of groups changes
  // (+ on resize), with a couple of delayed passes to catch entrance layout.
  const nGroups = stage.groups.length;
  const trackSig = stage.groups.map((g) => g.calls.length).join(",");
  useEffect(() => {
    const outer = trackRef.current, inner = innerRef.current;
    if (!outer || !inner) return;
    const compute = () => {
      const cw = outer.clientWidth, sw = inner.scrollWidth, pad = 44;
      const tx = Math.round(sw <= cw - pad * 2 ? (cw - sw) / 2 : cw - pad - sw);
      inner.style.transform = `translateX(${tx}px)`;
    };
    const ids = [setTimeout(compute, 30), setTimeout(compute, 340), setTimeout(compute, 640)];
    window.addEventListener("resize", compute);
    return () => { ids.forEach(clearTimeout); window.removeEventListener("resize", compute); };
  }, [nGroups, trackSig]);

  const active = shouldConnect && readyState === ReadyState.OPEN;
  const phase = stage.phase;
  const userActive = active && phase === "listening";
  const sagaActive = active && (phase === "thinking" || phase === "searching" || phase === "answering");

  // Dim tool groups by their rank among the turns present (newest = full),
  // robust to gaps in the conv counter.
  const convOrder = Array.from(new Set(stage.groups.map((g) => g.conv))).sort((a, b) => a - b);
  const rankAge = (conv: number) => convOrder.length - 1 - convOrder.indexOf(conv);

  if (!healthStatus || !backendServerUrl) {
    return (<div className="saga-boot"><SagaStyle /><div className="saga-mark" /><span>connecting to SAGA…</span></div>);
  }
  if (healthStatus && !healthStatus.ok) return <CouldNotConnect healthStatus={healthStatus} />;

  const renderCard = (c: ToolCall, i: number, answerText: string) => {
    const shown = c.papers?.slice(0, 2) ?? [];
    const more = (c.papers?.length ?? 0) - shown.length;
    return (
      <div key={(c.id || c.tool) + i} className={`saga-card saga-search ${c.status === "running" ? "searching" : "found"}`}>
        <div className="hd">
          <span className="dot" />
          <span className="tool">{c.tool}()</span>
          <span className="status">{c.error ? "error" : c.status === "running" ? "running" : (c.count ?? 0) + " papers"}</span>
        </div>
        <div className="q">{c.query || "…"}</div>
        {(c.freshness != null || c.impact != null) && (
          <div className="params">
            {c.freshness != null && (<span className="chip">freshness<span className="meter"><i style={{ width: `${c.freshness * 100}%` }} /></span><b>{c.freshness.toFixed(1)}</b></span>)}
            {c.impact != null && (<span className="chip">impact<span className="meter"><i style={{ width: `${c.impact * 100}%` }} /></span><b>{c.impact.toFixed(1)}</b></span>)}
          </div>
        )}
        {shown.length > 0 && (
          <div className="papers">
            {shown.map((p, j) => (
              <div key={j} className={`saga-paper ${citedInAnswer(p, answerText) ? "cited" : ""}`}>
                <div className="ttl">{p.title}</div>
                <div className="meta">
                  {p.authors && <span>{p.authors}</span>}
                  {p.year != null && <span className="yr">{p.year}</span>}
                  {p.citations != null && <span className="cit">{p.citations} cites</span>}
                </div>
              </div>
            ))}
            {more > 0 && <div className="more">+{more} more</div>}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="saga-stage" data-phase={phase}>
      <SagaStyle />
      <ErrorMessages errors={errors} setErrors={setErrors} />

      <header className="saga-brand">
        <div className="saga-mark" />
        <div><h1>SAGA</h1><p>spoken · agentic · grounded assistant</p></div>
      </header>

      <div className="saga-main">
        <div className="saga-stage-row">
          {(stage.user || stage.query) ? (
            <div className="saga-bubble user"><span className="who">You · asking</span>
              <p className="txt">{stage.query || stage.user}{!stage.query && <span className="caret">▍</span>}</p></div>
          ) : <div className="saga-slot" />}

          <div className="saga-orbs">
            <div className="saga-side">
              <div
                className={`saga-orb user ${userActive ? "on" : ""} ${!active ? "waiting" : ""}`}
                onClick={onConnectButtonPress}
                onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onConnectButtonPress(); } }}
                role="button"
                tabIndex={0}
                title={active ? "tap to end session" : "tap to start"}
              >
                <SagaOrb proc={audioProcessor} which="input" colorVar="--human" active={userActive} connected={active} />
                {!active && (
                  <span className="saga-play" aria-hidden="true">
                    <svg viewBox="0 0 24 24" width="38" height="38"><path d="M8 5l11 7-11 7z" fill="currentColor" /></svg>
                  </span>
                )}
              </div>
              <span className="saga-orb-label" style={{ color: "var(--human)" }}>{active ? "you" : "tap to talk"}</span>
            </div>
            <div className="saga-side">
              <div className={`saga-orb saga ${sagaActive ? "on" : ""} ${stage.speculating ? "anticipating" : ""}`}>
                <SagaOrb proc={audioProcessor} which="output" colorVar="--signal" active={sagaActive} connected={active} />
              </div>
              <span className="saga-orb-label" style={{ color: "var(--signal)" }}>SAGA</span>
              {(stage.speculating || stage.specNote) && (
                <div className={`saga-spec ${stage.specNote ? "committed" : "live"}`} title={stage.specHyp ? `predicted from: "${stage.specHyp}"` : undefined}>
                  <span className="bolt" aria-hidden="true" />
                  {stage.specNote
                    ? <span className="txt"><b>anticipated</b> · “{stage.specNote}”</span>
                    : <span className="txt"><b>anticipating</b><span className="ell">…</span></span>}
                </div>
              )}
            </div>
          </div>

          {stage.answer ? <AnswerBubble text={stage.answer} /> : <div className="saga-slot" />}
        </div>

        <div className={`saga-track ${stage.groups.length ? "has-tools" : ""}`} ref={trackRef}>
          <div className="saga-track-inner" ref={innerRef}>
            {stage.groups.map((g) => (
              <div
                className={`saga-group in ${rankAge(g.conv) === 0 ? "current" : "past"}`}
                key={g.key}
                style={{ opacity: ageOpacity(rankAge(g.conv)) }}
              >
                {g.note && (<div className="saga-group-note"><span className="lbl">spoken note</span>{g.note}</div>)}
                <div className="saga-group-calls">{g.calls.map((c, i) => renderCard(c, i, stage.answer))}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {microphoneAccess === "refused" && (<div className="saga-mic-warn">Allow microphone access in your browser settings.</div>)}
    </div>
  );
};

const SagaStyle = () => <style>{CSS}</style>;
const CSS = `
:root{
  /* warm + light — near-white cream ground, warm accents. Pinned light so it
     stays creamish regardless of the viewer's OS theme (this is a demo stage). */
  color-scheme:light;
  --ground:#fbf8f2;--ground-2:#ffffff;--panel:rgba(255,255,255,.86);
  --ink:#2c2318;--muted:#8b7b66;--faint:#cdbfa9;--line:#efe7d8;
  --signal:#c97e1e;--signal-soft:#ecd8b4;--signal-glow:rgba(201,126,30,.26);
  --human:#b25a3e;--human-soft:#e6c7b6;
  --q-bg:rgba(201,126,30,.08);--agent-bg2:rgba(250,243,232,.55);
  --user-bg1:rgba(178,90,62,.11);--user-bg2:rgba(178,90,62,.04);
  --shadow:0 16px 40px -22px rgba(120,85,45,.32);
  --sans:"Segoe UI",system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
}
.saga-stage,.saga-boot{position:fixed;inset:0;overflow:hidden;color:var(--ink);font-family:var(--sans);
  background:radial-gradient(120% 90% at 50% 6%,var(--ground-2),var(--ground) 60%)}
.saga-boot{display:flex;flex-direction:column;gap:16px;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:13px;color:var(--muted);letter-spacing:.08em}
.saga-mark{width:34px;height:34px;border-radius:50%;position:relative;
  background:conic-gradient(from 210deg,var(--signal),transparent 55%,var(--human));box-shadow:0 0 18px -2px var(--signal-glow)}
.saga-mark::after{content:"";position:absolute;inset:7px;border-radius:50%;background:var(--ground)}
.saga-brand{position:absolute;top:22px;left:28px;z-index:6;display:flex;gap:12px;align-items:center}
.saga-brand h1{font-size:15px;letter-spacing:.16em;font-weight:600}
.saga-brand p{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.03em;margin-top:2px}

/* Main = column: persistent orb-stage over the tool track; they can't overlap. */
.saga-main{position:absolute;left:0;right:0;top:88px;bottom:34px;z-index:3;display:flex;flex-direction:column;gap:6px}
/* 3-column grid keeps the two orbs dead-center; transcripts fill the side columns. */
.saga-stage-row{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:1fr auto 1fr;
  align-items:center;gap:clamp(10px,2.5vw,34px);padding:0 3vw}
.saga-slot{min-width:0}

.saga-orbs{grid-column:2;display:flex;align-items:center;gap:clamp(8px,1.6vw,24px);justify-self:center}
.saga-side{display:flex;flex-direction:column;align-items:center;gap:9px;position:relative}
.saga-orb{position:relative;transition:transform .4s cubic-bezier(.2,.8,.2,1)}
.saga-orb.user{width:clamp(94px,11vw,140px);height:clamp(94px,11vw,140px);cursor:pointer}
.saga-orb.saga{width:clamp(116px,14vw,182px);height:clamp(116px,14vw,182px)}
.saga-orb.on{transform:scale(1.05)}
.saga-orb.user:focus-visible{outline:2px solid var(--human);outline-offset:5px;border-radius:50%}
.saga-orb-canvas{display:block;width:100%;height:100%}
.saga-orb-label{font-family:var(--mono);font-size:10px;letter-spacing:.18em;text-transform:uppercase;opacity:.85}

/* Speculation indicator — SAGA anticipated the turn and pre-synthesized a filler.
   Floats just under the SAGA orb so it never shifts the orb's centering. */
.saga-orb.saga.anticipating{animation:antic 1.5s ease-in-out infinite}
@keyframes antic{0%,100%{transform:scale(1)}50%{transform:scale(1.035)}}
.saga-orb.saga.anticipating .saga-orb-canvas{filter:drop-shadow(0 0 14px var(--signal-glow))}
/* Narrow + 2-line wrap so a long committed note grows DOWNWARD, not sideways
   into the answer ("system speech") column -- the SAGA orb leans right, so a
   wide centered pill used to overlap it. */
.saga-spec{position:absolute;top:calc(100% + 8px);left:50%;transform:translateX(-50%);z-index:6;
  display:flex;align-items:flex-start;gap:8px;width:max-content;max-width:min(34vw,250px);
  padding:6px 13px;border-radius:14px;font-family:var(--mono);font-size:11px;letter-spacing:.02em;
  background:var(--q-bg);border:1px solid var(--signal-soft);color:var(--signal);
  box-shadow:0 10px 26px -16px var(--signal-glow);animation:rise .32s cubic-bezier(.2,.8,.2,1)}
.saga-spec .bolt{margin-top:1px}
.saga-spec .txt{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;
  overflow:hidden;line-height:1.3}
.saga-spec .txt b{font-weight:600;letter-spacing:.09em;text-transform:uppercase;font-size:9.5px}
.saga-spec.committed{color:var(--ink);border-color:var(--line);background:var(--agent-bg2)}
.saga-spec.committed b{color:var(--signal)}
.saga-spec .bolt{width:9px;height:13px;flex:none;background:currentColor;
  clip-path:polygon(58% 0,8% 60%,44% 60%,40% 100%,92% 40%,54% 40%)}
.saga-spec.live .bolt{animation:boltpulse 1s ease-in-out infinite}
@keyframes boltpulse{0%,100%{opacity:.35}50%{opacity:1}}
.saga-spec .ell{animation:blink 1.1s steps(1) infinite}

/* Play affordance inside the "you" orb — pulses while waiting to be tapped. */
.saga-orb.waiting{animation:orbwait 1.9s ease-in-out infinite}
@keyframes orbwait{0%,100%{transform:scale(1)}50%{transform:scale(1.06)}}
.saga-play{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:2;color:var(--human);pointer-events:none}
.saga-play::before{content:"";position:absolute;width:52%;height:52%;border-radius:50%;background:rgba(255,251,246,.94);box-shadow:0 6px 18px -6px rgba(120,85,45,.55)}
.saga-play svg{position:relative;z-index:1;margin-left:3px;animation:playpulse 1.9s ease-in-out infinite}
@keyframes playpulse{0%,100%{opacity:.85;transform:scale(1)}50%{opacity:1;transform:scale(1.1)}}

.saga-bubble{padding:14px 19px;border-radius:18px;box-shadow:var(--shadow);animation:rise .5s cubic-bezier(.2,.8,.2,1);max-height:100%;overflow-y:auto}
.saga-bubble .who{font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;display:block;margin-bottom:6px}
.saga-bubble .txt{font-size:17px;line-height:1.5;min-height:1.2em}
.saga-bubble.user{grid-column:1;justify-self:end;max-width:min(420px,100%);background:linear-gradient(180deg,var(--user-bg1),var(--user-bg2));border:1px solid var(--human-soft);border-bottom-right-radius:6px}
.saga-bubble.user .who{color:var(--human)}
.saga-bubble.agent{grid-column:3;justify-self:start;max-width:min(520px,100%);backdrop-filter:blur(10px);background:linear-gradient(180deg,var(--panel),var(--agent-bg2));border:1px solid var(--line);border-bottom-left-radius:6px}
.saga-bubble.agent .who{color:var(--signal)}
.caret{display:inline-block;color:var(--signal);animation:blink 1s steps(1) infinite;margin-left:1px}
@keyframes blink{50%{opacity:0}}
@keyframes rise{from{opacity:0;transform:translateY(12px) scale(.98)}to{opacity:1;transform:none}}

/* Tool track: horizontal conveyor. Groups persist across turns (dim by age),
   newest slides in from the right, oldest off the left. Opens from 0 height. */
.saga-track{flex:0 0 auto;height:0;position:relative;overflow:hidden;
  transition:height .45s cubic-bezier(.3,.8,.3,1);
  -webkit-mask-image:linear-gradient(90deg,transparent,#000 5%,#000 95%,transparent);
  mask-image:linear-gradient(90deg,transparent,#000 5%,#000 95%,transparent)}
.saga-track.has-tools{height:min(330px,40vh)}
.saga-track-inner{position:absolute;top:0;left:0;display:inline-flex;gap:20px;align-items:flex-start;padding:6px 0;
  transition:transform .6s cubic-bezier(.25,.9,.3,1);will-change:transform}
.saga-group{display:flex;flex-direction:column;gap:10px;max-width:540px;
  transition:opacity .5s ease,filter .5s ease,transform .5s ease}
.saga-group.in{animation:groupin .5s ease}
/* Past turns recede: desaturated, slightly shrunk, so the live (current) turn's
   tool calls stay the clear focal point and interruption mid-turn reads cleanly. */
.saga-group.past{filter:grayscale(.62) saturate(.7);transform:scale(.965)}
.saga-group.current{filter:none}
@keyframes groupin{from{opacity:0;transform:translateX(24px)}to{transform:none}}
.saga-group-note{font-family:var(--mono);font-size:12.5px;line-height:1.45;color:var(--ink);
  background:var(--q-bg);border:1px solid var(--signal-soft);border-left:3px solid var(--signal);
  border-radius:11px;padding:8px 13px}
.saga-group-note .lbl{display:block;font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--signal);opacity:.85;margin-bottom:3px}
.saga-group-calls{display:flex;gap:12px;align-items:flex-start}

.saga-card{width:250px;flex:0 0 250px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px 13px;backdrop-filter:blur(12px);box-shadow:var(--shadow)}
.saga-card .hd{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.saga-card .dot{width:7px;height:7px;border-radius:50%;background:var(--signal);box-shadow:0 0 8px var(--signal-glow);flex:0 0 auto}
.saga-card .tool{font-family:var(--mono);font-size:12px;color:var(--signal);font-weight:600}
.saga-card .status{margin-left:auto;font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.saga-search .q{font-family:var(--mono);font-size:12px;color:var(--ink);line-height:1.4;padding:7px 9px;border-radius:8px;background:var(--q-bg);border:1px solid var(--signal-soft)}
.saga-search .q::before{content:"\\201C";color:var(--faint)}.saga-search .q::after{content:"\\201D";color:var(--faint)}
.params{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.chip{font-family:var(--mono);font-size:10px;color:var(--muted);border:1px solid var(--line);border-radius:6px;padding:3px 7px;display:flex;gap:6px;align-items:center}
.chip b{color:var(--ink)}
.meter{width:30px;height:4px;border-radius:3px;background:var(--line);overflow:hidden;position:relative;display:inline-block}
.meter i{position:absolute;inset:0 auto 0 0;background:var(--signal)}
.searching{border-color:var(--signal)}
.searching .dot{animation:pulse 1.1s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 var(--signal-glow)}50%{box-shadow:0 0 0 6px transparent;opacity:.5}}
.found .status{color:var(--signal)}
.papers{display:flex;flex-direction:column;gap:8px;margin-top:10px}
.saga-paper{border:1px solid var(--line);border-radius:11px;padding:9px 11px;background:rgba(0,0,0,.03)}
.saga-paper .ttl{font-size:12.5px;line-height:1.32;font-weight:600;color:var(--ink)}
.saga-paper .meta{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:5px;display:flex;gap:8px;flex-wrap:wrap}
.saga-paper .yr{color:var(--signal)}.saga-paper .cit{color:var(--human)}
.saga-paper.cited{border-color:var(--signal);box-shadow:0 0 0 1px var(--signal),0 0 24px -6px var(--signal-glow)}
.saga-paper.cited .ttl{color:var(--signal)}
.saga-card .more{font-family:var(--mono);font-size:10px;color:var(--muted);padding-left:2px}

.saga-mic-warn{position:absolute;left:50%;bottom:20px;transform:translateX(-50%);z-index:6;color:#f4a;font-size:12px}
@media (max-width:960px){.saga-card{width:210px;flex-basis:210px}.saga-track.has-tools{height:min(290px,38vh)}.saga-group{max-width:460px}
  .saga-bubble{font-size:15px}}
@media (prefers-reduced-motion:reduce){*{animation-duration:.001s!important}.caret{display:none}.saga-track-inner,.saga-orb{transition:none}}
`;

export default Saga;
