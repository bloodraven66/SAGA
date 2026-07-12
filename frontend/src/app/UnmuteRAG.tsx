"use client";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { useCallback, useEffect, useRef, useState } from "react";
import { useMicrophoneAccess } from "./useMicrophoneAccess";
import { base64DecodeOpus, base64EncodeOpus } from "./audioUtil";
import { useAudioProcessor as useAudioProcessor } from "./useAudioProcessor";
import useKeyboardShortcuts from "./useKeyboardShortcuts";
import PositionedAudioVisualizer from "./PositionedAudioVisualizer";
import CouldNotConnect, { HealthStatus } from "./CouldNotConnect";
import Subtitles from "./Subtitles";
import { Frank_Ruhl_Libre, Raleway } from "next/font/google";
import fitLogo from "../assets/but-speech-at-fit-logo.png";
import { ChatMessage, compressChatHistory } from "./chatHistory";
import useWakeLock from "./useWakeLock";
import ErrorMessages, { ErrorItem, makeErrorItem } from "./ErrorMessages";
import clsx from "clsx";
import { useBackendServerUrl } from "./useBackendServerUrl";
import { RECORDING_CONSENT_STORAGE_KEY } from "./ConsentModal";

const RAG_VOICE = "unmute-prod-website/p329_022.wav"; // Watercooler

const frankRuhlLibre = Frank_Ruhl_Libre({ weight: "400", subsets: ["latin"] });
const raleway = Raleway({ weight: ["300", "600"], subsets: ["latin"] });

const RagHeader = () => (
  <div className="flex flex-col items-end gap-1 py-4 md:py-6 select-none">
    <div className={`text-3xl tracking-widest font-semibold text-gray-800 ${raleway.className}`}>
      FIT-Voice
    </div>
    <div className={`text-xs text-gray-400 ${frankRuhlLibre.className}`}>
      built on Kyutai / Unmute
    </div>
  </div>
);

type RagCall = {
  generation_index: number;
  rag_input_query: string;
  rag_context_injected: string;
  rag_applied_to_prompt: boolean;
  rag_not_applied_reason: string | null;
  rag_latency_ms: number | null;
};

type RagEntry = {
  call: RagCall;
  answer: string;
};

/** Join the last consecutive assistant deltas without newlines. */
const getLastAssistantRun = (history: ChatMessage[]): string => {
  const run: string[] = [];
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].role === "assistant") run.unshift(history[i].content);
    else break;
  }
  return run.join("").trim();
};

const RagEntryCard = ({
  entry,
  live,
  liveAnswer,
}: {
  entry: RagEntry;
  live: boolean;
  liveAnswer: string;
}) => {
  const answer = live ? liveAnswer : entry.answer;
  const { call } = entry;

  return (
    <div className="border border-black/10 rounded-lg overflow-hidden text-xs font-mono text-black/75 shrink-0">
      <div className="bg-black/5 px-3 py-1.5 flex items-center gap-2 text-[10px] uppercase tracking-widest text-black/35">
        <span>#{call.generation_index + 1}</span>
        {call.rag_latency_ms != null && (
          <span>{Math.round(call.rag_latency_ms)} ms</span>
        )}
        {!call.rag_applied_to_prompt && call.rag_not_applied_reason && (
          <span className="text-amber-700/70">{call.rag_not_applied_reason}</span>
        )}
        {live && <span className="ml-auto text-emerald-700/60 animate-pulse">live</span>}
      </div>

      <div className="divide-y divide-black/8">
        <div className="px-3 py-2">
          <div className="text-black/35 mb-1">Question</div>
          <div className="whitespace-pre-wrap break-words">
            {call.rag_input_query || <span className="text-black/20 italic">—</span>}
          </div>
        </div>

        <div className="px-3 py-2">
          <div className="text-black/35 mb-1">Retrieved context</div>
          {call.rag_applied_to_prompt && call.rag_context_injected ? (
            <div className="whitespace-pre-wrap break-words">{call.rag_context_injected}</div>
          ) : (
            <div className="text-black/20 italic">no context injected</div>
          )}
        </div>

        <div className="px-3 py-2">
          <div className="text-black/35 mb-1">Answer</div>
          <div className="whitespace-pre-wrap break-words">
            {answer || <span className="text-black/20 italic">—</span>}
          </div>
        </div>
      </div>
    </div>
  );
};

const UnmuteRAG = () => {
  const { showSubtitles } = useKeyboardShortcuts();
  const [ragEntries, setRagEntries] = useState<RagEntry[]>([]);
  const currentGenIndexRef = useRef<number>(-1);
  const voice = RAG_VOICE;
  const [rawChatHistory, setRawChatHistory] = useState<ChatMessage[]>([]);
  const rawChatHistoryRef = useRef<ChatMessage[]>([]);
  rawChatHistoryRef.current = rawChatHistory;
  const chatHistory = compressChatHistory(rawChatHistory);

  const { microphoneAccess, askMicrophoneAccess } = useMicrophoneAccess();

  const [shouldConnect, setShouldConnect] = useState(false);
  const backendServerUrl = useBackendServerUrl();
  const [webSocketUrl, setWebSocketUrl] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [errors, setErrors] = useState<ErrorItem[]>([]);
  const ragPanelRef = useRef<HTMLDivElement>(null);

  useWakeLock(shouldConnect);

  // Override circle colors for the light-themed RAG page
  useEffect(() => {
    const root = document.documentElement;
    const prevGreen = root.style.getPropertyValue("--color-green");
    const prevWhite = root.style.getPropertyValue("--color-white");
    root.style.setProperty("--color-green", "#B8955A"); // warm amber (assistant)
    root.style.setProperty("--color-white", "#9B6B5E"); // warm terracotta (user)
    return () => {
      root.style.setProperty("--color-green", prevGreen);
      root.style.setProperty("--color-white", prevWhite);
    };
  }, []);


  useEffect(() => {
    if (!backendServerUrl) return;

    setWebSocketUrl(backendServerUrl.toString() + "/v1/realtime");

    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        const response = await fetch(`${backendServerUrl}/v1/health`, {
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (!response.ok) {
          setHealthStatus({ connected: "yes_request_fail", ok: false });
          return;
        }
        const data = await response.json();
        data["connected"] = "yes_request_ok";
        setHealthStatus(data);
      } catch {
        setHealthStatus({ connected: "no", ok: false });
      }
    };

    checkHealth();
  }, [backendServerUrl]);

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    webSocketUrl || null,
    { protocols: ["realtime"] },
    shouldConnect
  );

  const onOpusRecorded = useCallback(
    (opus: Uint8Array) => {
      sendMessage(
        JSON.stringify({
          type: "input_audio_buffer.append",
          audio: base64EncodeOpus(opus),
        })
      );
    },
    [sendMessage]
  );

  const { setupAudio, shutdownAudio, audioProcessor } =
    useAudioProcessor(onOpusRecorded);

  const onConnectButtonPress = async () => {
    if (!shouldConnect) {
      const mediaStream = await askMicrophoneAccess();
      if (mediaStream) {
        await setupAudio(mediaStream);
        setShouldConnect(true);
      }
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
    if (lastMessage === null) return;

    const data = JSON.parse(lastMessage.data);
    if (data.type === "response.audio.delta") {
      const opus = base64DecodeOpus(data.delta);
      const ap = audioProcessor.current;
      if (!ap) return;
      ap.decoder.postMessage({ command: "decode", pages: opus }, [opus.buffer]);
    } else if (data.type === "unmute.additional_outputs") {
      const ragCall: RagCall | undefined = data.args?.debug_dict?.last_rag_call;
      if (!ragCall) return;

      const genIdx = ragCall.generation_index;
      if (genIdx !== currentGenIndexRef.current) {
        currentGenIndexRef.current = genIdx;
        const frozenAnswer = getLastAssistantRun(rawChatHistoryRef.current);
        setRagEntries((prev) => {
          const updated =
            prev.length > 0
              ? [...prev.slice(0, -1), { ...prev[prev.length - 1], answer: frozenAnswer }]
              : prev;
          return [...updated, { call: ragCall, answer: "" }];
        });
      } else {
        setRagEntries((prev) =>
          prev.length > 0
            ? [...prev.slice(0, -1), { ...prev[prev.length - 1], call: ragCall }]
            : prev
        );
      }
    } else if (data.type === "error") {
      if (data.error.type !== "warning") {
        setErrors((prev) => [...prev, makeErrorItem(data.error.message)]);
      }
    } else if (data.type === "conversation.item.input_audio_transcription.delta") {
      setRawChatHistory((prev) => [...prev, { role: "user", content: data.delta }]);
    } else if (data.type === "response.text.delta") {
      setRawChatHistory((prev) => [...prev, { role: "assistant", content: " " + data.delta }]);
    }
  }, [audioProcessor, lastMessage]);

  useEffect(() => {
    if (readyState !== ReadyState.OPEN) return;

    const recordingConsent =
      localStorage.getItem(RECORDING_CONSENT_STORAGE_KEY) === "true";

    setRawChatHistory([]);
    setRagEntries([]);
    currentGenIndexRef.current = -1;
    sendMessage(
      JSON.stringify({
        type: "session.update",
        session: {
          voice,
          allow_recording: recordingConsent,
        },
      })
    );
  }, [voice, readyState, sendMessage]);

  useEffect(() => {
    const el = ragPanelRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [ragEntries.length]);

  const liveAnswer = getLastAssistantRun(rawChatHistory);

  if (!healthStatus || !backendServerUrl) {
    return (
      <div className="flex flex-col gap-4 items-center bg-[#f0eeea] w-screen h-screen justify-center">
        <h1 className="text-xl text-gray-600">Loading...</h1>
      </div>
    );
  }

  if (healthStatus && !healthStatus.ok) {
    return <CouldNotConnect healthStatus={healthStatus} />;
  }

  return (
    <div className="w-screen h-screen flex overflow-hidden bg-[#f0eeea]">
      <ErrorMessages errors={errors} setErrors={setErrors} />

      {/* Left: audio + controls */}
      <div className="flex flex-col flex-1 min-w-0 items-center overflow-hidden">
        <header className="w-full px-3 md:px-6 flex justify-between items-center z-10 shrink-0">
          <img src={fitLogo.src} alt="BUT Speech@FIT" className="h-14 md:h-16 object-contain" />
          <RagHeader />
        </header>

        <div
          className={clsx(
            "w-full flex-1 min-h-0",
            "flex flex-row-reverse md:flex-row items-center justify-center",
            "md:-mr-4"
          )}
        >
          <PositionedAudioVisualizer
            chatHistory={chatHistory}
            role={"assistant"}
            analyserNode={audioProcessor.current?.outputAnalyser || null}
            onCircleClick={onConnectButtonPress}
            isConnected={shouldConnect}
          />
          <PositionedAudioVisualizer
            chatHistory={chatHistory}
            role={"user"}
            analyserNode={audioProcessor.current?.inputAnalyser || null}
            isConnected={shouldConnect}
          />
        </div>

        {showSubtitles && <Subtitles chatHistory={chatHistory} />}

        {microphoneAccess === "refused" && (
          <div className="text-red-600 text-sm text-center mb-4 shrink-0">
            Allow microphone access in your browser settings.
          </div>
        )}
      </div>

      {/* Right: RAG panel */}
      <div
        ref={ragPanelRef}
        className="w-96 shrink-0 border-l border-black/10 overflow-y-auto flex flex-col gap-3 p-3"
      >
        {ragEntries.length === 0 && (
          <div className="text-black/20 text-xs font-mono text-center mt-8 uppercase tracking-widest">
            RAG responses
          </div>
        )}
        {ragEntries.map((entry, i) => {
          const isLast = i === ragEntries.length - 1;
          return (
            <RagEntryCard
              key={entry.call.generation_index}
              entry={entry}
              live={isLast}
              liveAnswer={isLast ? liveAnswer : entry.answer}
            />
          );
        })}
      </div>
    </div>
  );
};

export default UnmuteRAG;
