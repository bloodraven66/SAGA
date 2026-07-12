import type { Metadata } from "next";
import Saga from "../Saga";

export const metadata: Metadata = {
  title: "SAGA — Spoken Agentic Grounded Assistant",
};

export default function SagaPage() {
  return <Saga />;
}
