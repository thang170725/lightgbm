export type ModelChoice = "lightgbm" | "ridge";

const API_BASE = "http://localhost:8000";

const endpoints: Record<ModelChoice, string> = {
    lightgbm: `${API_BASE}/predict_lightgbm`,
    ridge: `${API_BASE}/predict_linear`,
};

export type PredictResponse = {
    result: { time: string; value: number }[];
    evaluate: {
        mae: number;
        relative_mae?: number;
        mape: number;
        rmse: number;
        r2: number;
    };
    error?: string;
};

export async function predictFile(
    file: File,
    model: ModelChoice
): Promise<PredictResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(endpoints[model], {
        method: "POST",
        body: formData,
    });

    const data: PredictResponse = await res.json();

    if (!res.ok) {
        throw new Error(data.error ?? "Request failed");
    }

    if (data.error) {
        throw new Error(data.error);
    }

    return data;
}
