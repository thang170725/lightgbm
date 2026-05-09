"use client";

type DataPoint = {
    time: string;
    value: number;
};

export default function Table({
    data
}: {
    data: DataPoint[]
}) {

    return (
        <div className="
            bg-white
            rounded-2xl
            shadow-sm
            border
            overflow-hidden
        ">
            <div className="p-6">
                <h2 className="
                    text-xl
                    font-semibold
                ">
                    Forecast Table
                </h2>
            </div>

            <div className="overflow-x-auto">
                <table className="
                    min-w-full
                    text-sm
                ">
                    <thead className="
                        bg-gray-100
                        text-gray-700
                    ">
                        <tr>
                            <th className="
                                px-6
                                py-4
                                text-left
                            ">
                                Time
                            </th>

                            <th className="
                                px-6
                                py-4
                                text-left
                            ">
                                Prediction
                            </th>
                        </tr>
                    </thead>

                    <tbody>
                        {data.map((row, idx) => (
                            <tr
                                key={idx}
                                className="
                                    border-t
                                    hover:bg-gray-50
                                "
                            >
                                <td className="
                                    px-6
                                    py-4
                                ">
                                    {row.time}
                                </td>

                                <td className="
                                    px-6
                                    py-4
                                    font-medium
                                ">
                                    {row.value.toFixed(4)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}