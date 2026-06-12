"use client";

import { useState } from "react";

import type { EvaluateMetrics } from "@/src/components/Metrics";
import {
    predictFile,
    type ModelChoice,
} from "@/routes/UploadApi";

type ForecastPoint = { time: string; value: number };

type Props = {
    setForecast: (rows: ForecastPoint[]) => void;
    setEvaluate: (m: EvaluateMetrics | null) => void;
    setActiveModel: (m: ModelChoice | null) => void;
};

const MODEL_OPTIONS: {
    id: ModelChoice;
    label: string;
    description: string;
}[] = [
    {
        id: "lightgbm",
        label: "LightGBM",
        description: "lightgbm_model_v2.pkl",
    },
    {
        id: "ridge",
        label: "Ridge",
        description: "linear_v2.pkl",
    },
];

export default function Upload({
    setForecast,
    setEvaluate,
    setActiveModel,
}: Props) {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [model, setModel] = useState<ModelChoice>("lightgbm");

    const handleUpload = async () => {
        if (!file) return;

        try {
            setLoading(true);

            const data = await predictFile(file, model);

            setForecast(data.result);
            setEvaluate(data.evaluate);
            setActiveModel(model);
        } catch (err) {
            console.error(err);
            const msg =
                err instanceof Error ? err.message : "Upload failed!";
            alert(msg);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full shrink-0 lg:max-w-lg">
            <div className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-[0_4px_24px_-4px_rgba(15,23,42,0.08)]">
                <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Model & data
                </p>

                <div
                    className="mb-4 grid grid-cols-2 gap-2 rounded-xl bg-slate-100/80 p-1"
                    role="group"
                    aria-label="Select prediction model"
                >
                    {MODEL_OPTIONS.map((opt) => {
                        const selected = model === opt.id;
                        return (
                            <button
                                key={opt.id}
                                type="button"
                                onClick={() => setModel(opt.id)}
                                disabled={loading}
                                className={[
                                    "rounded-lg px-3 py-2.5 text-left transition",
                                    selected
                                        ? opt.id === "lightgbm"
                                            ? "bg-white text-teal-900 shadow-sm ring-1 ring-teal-600/20"
                                            : "bg-white text-indigo-900 shadow-sm ring-1 ring-indigo-600/20"
                                        : "text-slate-600 hover:text-slate-900",
                                    loading ? "cursor-not-allowed opacity-60" : "",
                                ].join(" ")}
                            >
                                <span className="block text-sm font-semibold">
                                    {opt.label}
                                </span>
                                <span className="mt-0.5 block text-[10px] text-slate-500">
                                    {opt.description}
                                </span>
                            </button>
                        );
                    })}
                </div>

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
                        className={[
                            "inline-flex items-center justify-center rounded-xl px-6 py-3 text-sm font-semibold text-white shadow-md transition disabled:cursor-not-allowed disabled:opacity-45",
                            model === "lightgbm"
                                ? "bg-linear-to-br from-teal-600 to-teal-700 shadow-teal-900/15 hover:from-teal-500 hover:to-teal-600"
                                : "bg-linear-to-br from-indigo-600 to-indigo-700 shadow-indigo-900/15 hover:from-indigo-500 hover:to-indigo-600",
                        ].join(" ")}
                    >
                        {loading ? (
                            <span className="flex items-center gap-2">
                                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                                Predicting…
                            </span>
                        ) : (
                            `Run ${model === "lightgbm" ? "LightGBM" : "Ridge"}`
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
