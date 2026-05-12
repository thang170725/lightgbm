"use client";

import { useState } from "react";
import Upload from "@/src/components/Upload";
import Chart from "@/src/components/Chart";
import Table from "@/src/components/Table";
import Metrics, { type EvaluateMetrics } from "@/src/components/Metrics";

type ForecastPoint = { time: string; value: number };

export default function Home() {
    const [forecast, setForecast] = useState<ForecastPoint[]>([]);
    const [evaluate, setEvaluate] = useState<EvaluateMetrics | null>(null);

    const hasResults = forecast.length > 0 && evaluate !== null;

    return (
        <main className="page-surface min-h-screen">
            <div className="mx-auto max-w-350 px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
                <header className="mb-8 flex flex-col gap-6 lg:mb-10 lg:flex-row lg:items-end lg:justify-between">
                    <div className="max-w-xl">
                        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-teal-700/90">
                            LightGBM · hourly forecast
                        </p>
                        <h1 className="text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
                            Electricity load prediction
                        </h1>
                        <p className="mt-3 text-base leading-relaxed text-slate-600">
                            Upload historical CSV (<code className="rounded bg-slate-200/80 px-1.5 py-0.5 text-sm text-slate-800">time</code>,{" "}
                            <code className="rounded bg-slate-200/80 px-1.5 py-0.5 text-sm text-slate-800">value</code>) to get the next 24h forecast and model metrics.
                        </p>
                    </div>
                    <Upload setForecast={setForecast} setEvaluate={setEvaluate} />
                </header>

                {hasResults && (
                    <div className="space-y-6">
                        <Metrics evaluate={evaluate} />

                        <div className="grid gap-6 lg:grid-cols-12 lg:items-start">
                            <section className="lg:col-span-8">
                                <Chart data={forecast} />
                            </section>
                            <aside className="flex flex-col gap-6 lg:col-span-4">
                                <Table data={forecast} />
                            </aside>
                        </div>
                    </div>
                )}

                {!hasResults && (
                    <div className="rounded-2xl border border-dashed border-slate-300/80 bg-white/60 px-6 py-12 text-center backdrop-blur-sm">
                        <p className="text-sm font-medium text-slate-700">
                            No forecast yet
                        </p>
                        <p className="mt-2 text-sm text-slate-500">
                            Choose a CSV file and run forecast to see the chart and detailed table.
                        </p>
                    </div>
                )}
            </div>
        </main>
    );
}