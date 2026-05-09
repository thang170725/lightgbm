"use client";

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    CartesianGrid,
} from "recharts";

type DataPoint = {
    time: string;
    value: number;
};

export default function Chart({
    data
}: {
    data: DataPoint[]
}) {

    return (
        <div className="
            bg-white
            p-6
            rounded-2xl
            shadow-sm
            border
            h-[450px]
            mb-8
        ">
            <h2 className="
                text-xl
                font-semibold
                mb-6
            ">
                Forecast Chart
            </h2>

            <ResponsiveContainer
                width="100%"
                height="100%"
            >
                <LineChart data={data}>
                    <CartesianGrid
                        strokeDasharray="3 3"
                    />

                    <XAxis
                        dataKey="time"
                        tickFormatter={(t) =>
                            t.slice(11, 16)
                        }
                    />

                    <YAxis />

                    <Tooltip />

                    <Line
                        type="monotone"
                        dataKey="value"
                        name="Prediction"
                        stroke="#2563eb"
                        strokeWidth={3}
                        dot={false}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}