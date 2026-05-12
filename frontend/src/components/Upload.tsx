"use client";

import { useState } from "react";

import type { EvaluateMetrics } from "@/src/components/Metrics";
import { predictFile } from "@/routes/UploadApi";

type ForecastPoint = { time: string; value: number };

type Props = {
    setForecast: (rows: ForecastPoint[]) => void;
    setEvaluate: (m: EvaluateMetrics | null) => void;
};

export default function Upload({ setForecast, setEvaluate }: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);

    const handleUpload = async () => {
        if (!file) return;

        try {
            setLoading(true);

            const data = await predictFile(file);

            setForecast(data.result);
            setEvaluate(data.evaluate);
        } catch (err) {
            console.error(err);
            alert("Upload failed!");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full shrink-0 lg:max-w-md">
            <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)]">
                <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Data input
                </p>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
                    <label className="relative flex flex-1 cursor-pointer flex-col justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50/80 px-4 py-3 transition hover:border-teal-400/60 hover:bg-teal-50/40">
                        <span className="text-xs font-medium text-slate-600">
                            CSV file
                        </span>
                        <span className="mt-1 truncate text-sm text-slate-900">
                            {file ? file.name : "Click to choose…"}
                        </span>
                        <input
                            type="file"
                            accept=".csv"
                            className="absolute inset-0 cursor-pointer opacity-0"
                            onChange={(e) =>
                                setFile(e.target.files?.[0] ?? null)
                            }
                        />
                    </label>

                    <button
                        type="button"
                        onClick={handleUpload}
                        disabled={loading || !file}
                        className="inline-flex items-center justify-center rounded-xl bg-linear-to-br from-teal-600 to-teal-700 px-6 py-3 text-sm font-semibold text-white shadow-md shadow-teal-900/15 transition hover:from-teal-500 hover:to-teal-600 disabled:cursor-not-allowed disabled:opacity-45"
                    >
                        {loading ? (
                            <span className="flex items-center gap-2">
                                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                                Predicting…
                            </span>
                        ) : (
                            "Run forecast"
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}