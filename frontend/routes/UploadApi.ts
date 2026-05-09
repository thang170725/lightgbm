import axios from "axios";

export async function predictFile(file: File) {
    const formData = new FormData();

    formData.append("file", file);

    const res = await axios.post(
        "http://localhost:8000/predict",
        formData
    );

    console.log(res.data)
    return res.data
}