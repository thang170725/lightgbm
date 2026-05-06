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

export default function Chart({ data }: { data: DataPoint[] }) {
  return (
    <div className="w-full h-[400px] mt-6">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />

          <XAxis
            dataKey="time"
            tickFormatter={(t) => t.slice(11, 16)}
          />

          <YAxis />
          <Tooltip />

          {/* chỉ 1 line prediction */}
          <Line
            type="monotone"
            dataKey="value"
            name="Prediction"
            stroke="#dc2626"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}