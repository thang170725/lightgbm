export async function predictFile(file: File) {
    const formData = new FormData();

    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (!res.ok) {
        throw new Error("lỗi load file")
    }

    console.log(data);

    return data;
}