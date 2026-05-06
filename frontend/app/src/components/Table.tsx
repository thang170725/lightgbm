"use client";

type DataPoint = {
  time: string;
  value: number;
};

export default function Table({ data }: { data: DataPoint[] }) {
  return (
    <div className="mt-10 overflow-x-auto">
      <table className="min-w-full border">
        <thead>
          <tr className="bg-gray-100">
            <th className="border px-4 py-2">Time</th>
            <th className="border px-4 py-2">Predicted Value</th>
          </tr>
        </thead>

        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              <td className="border px-4 py-2">
                {row.time}
              </td>
              <td className="border px-4 py-2">
                {row.value.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}