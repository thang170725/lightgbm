"use client";

import { useState } from "react";

import Upload from "../src/components/Upload";
import Chart from "../src/components/Chart";
import Table from "../src/components/Table";
import Metrics from "@/src/components/Metrics";

export default function Home() {
    const [forecast, setForecast] = useState([]);

    const [evaluate, setEvaluate] = useState<any>(null);

    return (
        <main className="min-h-screen bg-gray-50 p-8">
            <div className="max-w-7xl mx-auto">

                <div className="mb-8">
                    <h1 className="
                        text-4xl
                        font-bold
                        text-gray-800
                    ">
                        Electricity Forecast ⚡
                    </h1>

                    <p className="text-gray-500 mt-2">
                        LightGBM electricity prediction dashboard
                    </p>
                </div>

                <Upload
                    setForecast={setForecast}
                    setEvaluate={setEvaluate}
                />

                {forecast.length > 0 && (
                    <>
                        <Metrics evaluate={evaluate} />

                        <Chart data={forecast} />

                        <Table data={forecast} />
                    </>
                )}
            </div>
        </main>
    );
}