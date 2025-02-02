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
            print(f"Submitting sequence (Attempt {attempt}): Heavy={heavy_chain[:10]}..., Light={light_chain[:10]}...")
            await page.goto("https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap", timeout=60000)
            await page.wait_for_load_state("networkidle")

            # Fill in sequences
            await page.fill("textarea[name='hchain']", heavy_chain)
            await page.fill("textarea[name='lchain']", light_chain)

            # Submit form
            await page.click("button.btn")

            # Wait for results to load
            await page.wait_for_selector("table.table-results", timeout=120000)

            # Extract descriptors
            descriptors = {}
            rows = await page.query_selector_all("table.table-results tr")
            for row in rows[1:]:
                cells = await row.query_selector_all("td")
                if len(cells) == 2:
                    metric = await cells[0].inner_text()
                    value = await cells[1].inner_text()
                    descriptors[metric.strip()] = value.strip()

            print(f"✔ Successfully processed sequence: {heavy_chain[:10]}...")
            return heavy_chain, descriptors  # Return data on success

        except Exception as e:
            print(f"⚠ Error processing sequence (Attempt {attempt}): {e}")
            # Exponential backoff before retry
            await asyncio.sleep(2 ** attempt)

    print(f"❌ Failed to process sequence after 3 attempts: {heavy_chain[:10]}...")
    return heavy_chain, None  # Return None if all retries failed


async def save_results(results, output_csv):
    """
    Saves results in CSV format.
    """
    print("💾 Saving results...")

    df = pd.DataFrame.from_dict(results, orient="index")
    df.reset_index(inplace=True)
    df.to_csv(output_csv, index=False)

    print(f"📁 Results saved to:\n  CSV: {output_csv}")


async def process_sequences(
    input_file,
    output_csv
):
    """
    Processes all sequences from CSV with batching, concurrency, and retry logic.
    """
    print("🔄 Loading sequences from CSV...")
    df = pd.read_csv(input_file)

    results = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        tasks = []
        for idx, row in df.iterrows():
            heavy_chain = row["sequence_aa_heavy"]
            light_chain = row["sequence_aa_light"]

            # Add submission to the batch
            tasks.append(submit_sequence(page, heavy_chain, light_chain))

            # Run in batches of max_concurrent_requests
            if len(tasks) >= 1:
                print(f"🚀 Running task {idx+1}/{len(df)}...")
                batch_results = await asyncio.gather(*tasks)
                tasks = []  # Clear the task list after processing

                # Save successful results
                for hc, desc in batch_results:
                    if desc:
                        results[hc] = desc

                # Save progress
                await save_results(results, output_csv)

        # Process any remaining tasks
        if tasks:
            print(f"🚀 Running task {len(df)}/{len(df)}...")
            batch_results = await asyncio.gather(*tasks)

            for hc, desc in batch_results:
                if desc:
                    results[hc] = desc

            await save_results(results, output_csv)

        await browser.close()

    print("✅ Processing completed!")


def main():
    """
    Main entry point. Parses arguments, then:
      1) (Optional) you could run extra pre/post-processing here.
      2) Runs the TAP scraping pipeline using the input CSV of sequences.
      3) Saves results to CSV in a single pipeline script.
    """
    parser = argparse.ArgumentParser(description="Run TAP scraping with standardized I/O")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV containing 'sequence_aa_heavy' and 'sequence_aa_light' columns"
    )
    parser.add_argument(
        "--output",
        help="Path to the output CSV file"
    )
    args = parser.parse_args()

    # For now, just run the TAP scraping:
    asyncio.run(
        process_sequences(
            input_file=args.input,
            output_csv=args.output
        )
    )

    print("Done: The pipeline steps have completed successfully!")


if __name__ == "__main__":
    main()
