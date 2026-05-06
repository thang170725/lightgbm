"use client";

import { useState } from "react";
import axios from "axios";

export default function Upload({ setData }: any) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;

    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post(
        "http://localhost:8000/predict",
        formData
      );

      // ⚠️ CHẶN DATA RÁC
      setData(Array.isArray(res.data) ? res.data : []);

    } catch (err) {
      console.error("Upload error:", err);
      alert("Upload failed!");
      setData([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border rounded-xl">
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      <button
        onClick={handleUpload}
        disabled={loading}
        className="ml-2 px-4 py-2 bg-blue-500 text-white disabled:opacity-50"
      >
        {loading ? "Predicting..." : "Predict"}
      </button>
    </div>
  );
}