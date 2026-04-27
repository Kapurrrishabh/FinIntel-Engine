export const API_BASE = process.env.NEXT_PUBLIC_API_URL!;

export const fetcher = (path: string) =>
  fetch(`${API_BASE}${path}`).then((res) => res.json());