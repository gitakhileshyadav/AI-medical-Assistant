# main.py
import os
import argparse
import requests

DEFAULT_API = os.getenv("API_URL", "http://127.0.0.1:8000")


def post_analyze(session: requests.Session, api: str, query: str, image_path: str = None):
    data = {"query": query}
    files = None
    if image_path:
        files = {"image_file": open(image_path, "rb")}
        resp = session.post(f"{api}/analyze", data=data, files=files, timeout=120)
    else:
        resp = session.post(f"{api}/analyze", data=data, timeout=120)
    return resp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--image", help="Path to image file for first request")
    parser.add_argument("--query", required=True, help="Query to send")
    args = parser.parse_args()

    s = requests.Session()
    print(f"Using API: {args.api}")

    # First request (with image if provided)
    resp = post_analyze(s, args.api, args.query, image_path=args.image)
    if resp.status_code == 200:
        print("AI:", resp.json().get("answer"))
    else:
        print("Error:", resp.status_code, resp.text)
        return

    # Follow-ups interactively
    try:
        while True:
            q = input("Follow-up (blank to exit): ").strip()
            if not q:
                break
            r2 = post_analyze(s, args.api, q, image_path=None)
            if r2.status_code == 200:
                print("AI:", r2.json().get("answer"))
            else:
                print("Error:", r2.status_code, r2.text)
                break
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
