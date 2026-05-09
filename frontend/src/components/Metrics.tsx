"use client";

export default function Metrics({
    evaluate
}: any) {

    const metrics = [
        {
            title: "MAE",
            value: evaluate.mae.toFixed(2)
        },
        {
            title: "MAPE",
            value: `${evaluate.mape.toFixed(2)}%`
        },
        {
            title: "RMSE",
            value: evaluate.rmse.toFixed(2)
        },
        {
            title: "R²",
            value: evaluate.r2.toFixed(4)
        }
    ];

    return (
        <div className="
            grid
            grid-cols-1
            md:grid-cols-4
            gap-4
            mb-8
        ">
            {metrics.map((m) => (
                <div
                    key={m.title}
                    className="
                        bg-white
                        p-6
                        rounded-2xl
                        shadow-sm
                        border
                    "
                >
                    <p className="
                        text-sm
                        text-gray-500
                    ">
                        {m.title}
                    </p>

                    <h2 className="
                        text-3xl
                        font-bold
                        mt-2
                    ">
                        {m.value}
                    </h2>
                </div>
            ))}
        </div>
    );
}