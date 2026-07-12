import type { Metadata } from "next";
import UnmuteRAG from "../UnmuteRAG";

export const metadata: Metadata = {
  title: "FIT-Voice",
};

export default function RagPage() {
  return (
    <div className="w-full h-screen flex justify-center bg-background">
      <UnmuteRAG />
    </div>
  );
}
