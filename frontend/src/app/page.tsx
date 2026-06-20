"use client";

import { useState, useRef, useCallback } from "react";

export default function Home() {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [patientContext, setPatientContext] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<object | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }
    setSelectedFile(file);
    setError(null);
    setResult(null);
    setPreview(URL.createObjectURL(file));
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const analyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("patient_context", patientContext);

    try {
      const res = await fetch("http://localhost:8000/api/analyze-scan", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Analysis failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setPatientContext("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const copyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Syntax-highlight JSON string → HTML
  const highlight = (json: string) =>
    json
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(
        /("(\\u[\dA-Fa-f]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        (match) => {
          let cls = "text-sky-300"; // number
          if (/^"/.test(match)) {
            cls = /:$/.test(match) ? "text-cyan-300" : "text-amber-200"; // key : string value
          } else if (/true|false/.test(match)) {
            cls = "text-emerald-400";
          } else if (/null/.test(match)) {
            cls = "text-rose-400";
          }
          return `<span class="${cls}">${match}</span>`;
        }
      );

  return (
    <div className="min-h-screen bg-[#0d1117] flex flex-col font-mono text-sm text-slate-300">
      {/* Header */}
      <header className="border-b border-white/10 px-6 py-3 flex items-center justify-between bg-[#161b22]">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-xs font-bold text-white">
            E
          </div>
          <span className="text-white font-semibold tracking-tight text-base font-sans">
            Enervara
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-sans">
            Medical AI
          </span>
        </div>
        <span className="text-xs text-slate-500 flex items-center gap-1.5 font-sans">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
          Gemini 2.5 Flash
        </span>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-8 space-y-6">
        {/* Upload + Context row */}
        <div className="grid md:grid-cols-2 gap-4">
          {/* Upload */}
          <div
            onClick={() => !selectedFile && fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            className={`
              relative rounded-xl border-2 border-dashed transition-all overflow-hidden flex items-center justify-center
              ${dragActive ? "border-cyan-400 bg-cyan-500/5" : "border-white/10 hover:border-white/20"}
              ${selectedFile ? "cursor-default" : "cursor-pointer bg-white/[0.02] hover:bg-white/[0.04]"}
            `}
            style={{ minHeight: "200px" }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
            />
            {preview ? (
              <div className="relative w-full h-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={preview} alt="scan" className="w-full object-contain" style={{ maxHeight: "200px" }} />
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
                <div className="absolute bottom-2 left-3 right-3 flex justify-between items-center">
                  <span className="text-xs text-slate-300 truncate max-w-[160px]">{selectedFile?.name}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); reset(); }}
                    className="text-xs px-2.5 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-slate-300 font-sans"
                  >
                    Remove
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center p-6 space-y-2">
                <div className="text-3xl">🩻</div>
                <p className="text-slate-400 font-sans text-sm">Drop scan or <span className="text-cyan-400 underline">click to upload</span></p>
                <p className="text-slate-600 text-xs font-sans">X-Ray · MRI · CT · Ultrasound</p>
              </div>
            )}
          </div>

          {/* Context + Button */}
          <div className="flex flex-col gap-3">
            <div className="flex-1 rounded-xl border border-white/10 bg-white/[0.02] p-4 space-y-2">
              <label className="text-xs text-slate-500 font-sans uppercase tracking-wider">
                Patient Context <span className="normal-case">(optional)</span>
              </label>
              <textarea
                value={patientContext}
                onChange={(e) => setPatientContext(e.target.value)}
                placeholder="e.g. 67-year-old male, smoker, chest pain..."
                rows={5}
                className="w-full resize-none bg-transparent text-slate-300 placeholder:text-slate-700 focus:outline-none text-xs leading-relaxed"
              />
            </div>
            <button
              onClick={analyze}
              disabled={!selectedFile || loading}
              className={`
                py-3 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2 font-sans
                ${!selectedFile || loading
                  ? "bg-white/5 text-slate-600 cursor-not-allowed"
                  : "bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:opacity-90 shadow-lg shadow-blue-500/20"}
              `}
            >
              {loading ? (
                <>
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Analyzing…
                </>
              ) : "▶  Run Analysis"}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-900/10 px-4 py-3 text-rose-300 text-xs font-sans">
            ✗ {error}
          </div>
        )}

        {/* JSON Output */}
        {result && (
          <div className="rounded-xl border border-white/10 bg-[#161b22] overflow-hidden">
            {/* JSON toolbar */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/10 bg-[#1c2128]">
              <div className="flex items-center gap-2">
                <span className="text-slate-500 text-xs">POST</span>
                <span className="text-slate-400 text-xs">/api/analyze-scan</span>
                <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  200 OK
                </span>
              </div>
              <button
                onClick={copyJson}
                className="text-xs px-3 py-1 rounded-lg border border-white/10 hover:bg-white/5 transition-colors text-slate-400 hover:text-slate-200 font-sans flex items-center gap-1.5"
              >
                {copied ? "✓ Copied" : "Copy JSON"}
              </button>
            </div>

            {/* JSON body */}
            <pre
              className="overflow-auto p-5 text-xs leading-6 text-slate-300"
              style={{ maxHeight: "70vh" }}
              dangerouslySetInnerHTML={{
                __html: highlight(JSON.stringify(result, null, 2)),
              }}
            />
          </div>
        )}
      </main>
    </div>
  );
}
