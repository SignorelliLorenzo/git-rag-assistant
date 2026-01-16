import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GitRepo RAG Assistant",
  description: "Chat with your repositories using the local RAG pipeline.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
