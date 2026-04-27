export const fetcher = (url: string) => fetch(url).then((res) => res.json());
export const API_BASE = "http://127.0.0.1:8000";