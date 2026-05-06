"use client";

import { useState } from "react";
import Upload from "./src/components/Upload";
import Chart from "./src/components/Chart";
import Table from "./src/components/Table";

export default function Home() {
  const [data, setData] = useState([]);

  return (
    <main className="p-10">
      <h1 className="text-2xl font-bold mb-4">
        Electricity Forecast ⚡
      </h1>

      <Upload setData={setData} />

      {data.length > 0 && (
        <>
          <Chart data={data} />
          <Table data={data} />
        </>
      )}
    </main>
  );
}