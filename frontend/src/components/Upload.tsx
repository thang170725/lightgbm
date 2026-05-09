"use client";

import { useState } from "react";

import { predictFile } from "@/routes/UploadApi";

export default function Upload({
    setForecast,
    setEvaluate
}: any) {

    const [file, setFile] =
        useState<File | null>(null);

    const [loading, setLoading] =
        useState(false);

    const handleUpload = async () => {
        if (!file) return;

        try {
            setLoading(true);

            const data =
                await predictFile(file);

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
        <div className="
            bg-white
            p-6
            rounded-2xl
            shadow-sm
            border
            mb-8
        ">
            <div className="
                flex
                flex-col
                md:flex-row
                gap-4
                items-center
            ">
                <input
                    type="file"
                    accept=".csv"
                    className="
                        border
                        p-2
                        rounded-lg
                        w-full
                    "
                    onChange={(e) =>
                        setFile(
                            e.target.files?.[0] || null
                        )
                    }
                />

                <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="
                        px-6
                        py-2
                        rounded-lg
                        bg-blue-600
                        text-white
                        hover:bg-blue-700
                        transition
                        disabled:opacity-50
                    "
                >
                    {loading
                        ? "Predicting..."
                        : "Run Forecast"}
                </button>
            </div>
        </div>
    );
}