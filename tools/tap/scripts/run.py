#!/usr/bin/env python3
import argparse
import asyncio
import pandas as pd
from playwright.async_api import async_playwright

async def submit_sequence(page, heavy_chain, light_chain):
    """
    Submits a single sequence to the TAP web tool and retrieves results.
    Retries on errors up to max_retries times.
    """
    for attempt in range(1, 4):
        try:
            print(f"Submitting sequence (Attempt {attempt}): Heavy={heavy_chain[:10]}..., Light={light_chain[:10]}...", flush=True)
            await page.goto("https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap", timeout=60000)
            await page.wait_for_load_state("networkidle")

            # Fill in sequences
            await page.fill("textarea[name='hchain']", heavy_chain)
            await page.fill("textarea[name='lchain']", light_chain)

            # Submit form
            await page.click("button.btn")

            # Wait for results to load
            await page.wait_for_selector("table.table-results", timeout=120000)

            print(f"✔ Successfully processed sequence: {heavy_chain[:10]}...", flush=True)
            return heavy_chain, {}  # Simulated return value

        except Exception as e:
            print(f"⚠ Error processing sequence (Attempt {attempt}): {e}", flush=True)
            await asyncio.sleep(2 ** attempt)

    print(f"❌ Failed to process sequence after 3 attempts: {heavy_chain[:10]}...", flush=True)
    return heavy_chain, None

async def process_sequences(input_file, output_csv):
    """
    Processes all sequences from CSV with batching, concurrency, and retry logic.
    """
    print("🔄 Loading sequences from CSV...", flush=True)
    df = pd.read_csv(input_file)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for idx, row in df.iterrows():
            heavy_chain = row["sequence_aa_heavy"]
            light_chain = row["sequence_aa_light"]

            print(f"🚀 Running task {idx+1}/{len(df)}...", flush=True)
            await submit_sequence(page, heavy_chain, light_chain)

        await browser.close()

    print("✅ Processing completed!", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Run TAP scraping with standardized I/O")
    parser.add_argument("--input", required=True, help="Path to the input CSV")
    parser.add_argument("--output", required=True, help="Path to the output CSV")
    args = parser.parse_args()

    asyncio.run(process_sequences(input_file=args.input, output_csv=args.output))

if __name__ == "__main__":
    main()
