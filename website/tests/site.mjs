import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, readFile } from "node:fs/promises";
import { chromium, expect } from "@playwright/test";

const data = JSON.parse(
  await readFile(new URL("../src/results.json", import.meta.url)),
);
const allowed = new Set([
  "model",
  "gpu",
  "bi",
  "qnbi",
  "quality",
  "family",
  "size",
  "moe",
  "mode",
]);
for (const row of [
  ...data.workloads.flatMap((w) => w.rows),
  ...data.reasoning,
  ...Object.values(data.gpus).flat(),
]) {
  assert.ok(
    Object.keys(row).every((key) => allowed.has(key)),
    "Unexpected public result field",
  );
  assert.ok(row.bi > 0 && row.qnbi > 0 && row.quality > 0 && row.quality <= 1);
  assert.ok(Math.abs(row.bi / row.quality / row.qnbi - 1) < 1e-8);
}
assert.equal(data.workloads.length, 9);
assert.ok(
  !JSON.stringify(data).match(
    /\/local\/|summary_path|job_id|energy_joules|api_key/,
  ),
);

const server = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "preview",
    "--host",
    "127.0.0.1",
    "--port",
    "4179",
    "--strictPort",
    "--base",
    "/BIRDS/",
  ],
  { stdio: "pipe" },
);
let browser;
try {
  await new Promise((resolve, reject) => {
    server.stdout.on("data", (text) => {
      if (text.toString().includes("4179")) resolve();
    });
    server.stderr.on("data", (text) => process.stderr.write(text));
    server.on("error", reject);
    server.on("exit", (code) => reject(new Error(`Preview exited: ${code}`)));
  });
  browser = await chromium.launch({ headless: true });
  await mkdir("test-output", { recursive: true });
  for (const [name, width, height] of [
    ["desktop", 1440, 1000],
    ["mobile", 390, 844],
    ["small-mobile", 320, 720],
  ]) {
    const context = await browser.newContext({
      viewport: { width, height },
      reducedMotion: "reduce",
      hasTouch: width < 500,
    });
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("response", (response) => {
      if (response.status() >= 400)
        errors.push(`${response.status()} ${response.url()}`);
    });
    await page.goto("http://127.0.0.1:4179/BIRDS/", {
      waitUntil: "networkidle",
    });
    assert.equal(await page.locator("canvas").count(), 4);
    assert.equal(await page.locator(".source-row").count(), 12);
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "Page overflows horizontally",
    );
    assert.ok(
      await page.evaluate(async () => {
        const img = new Image();
        img.src = "/BIRDS/forest.jpg";
        await img.decode();
        return img.naturalWidth > 1000;
      }),
    );
    for (const canvas of await page.locator("canvas").all()) {
      const nonblank = await canvas.evaluate((el) => {
        const pixels = el
          .getContext("2d")
          .getImageData(0, 0, el.width, el.height).data;
        let count = 0;
        for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) count++;
        return count > 1000;
      });
      assert.ok(nonblank, "Blank canvas");
    }
    await page.screenshot({ path: `test-output/${name}.png`, fullPage: true });
    await page.selectOption("#composition-view", "midpoint");
    await page.selectOption("#horizon", "20");
    assert.match(
      await page.locator("#composition-table").textContent(),
      /71\.0984/,
    );
    for (const workload of data.workloads) {
      await page.selectOption("#workload", workload.key);
      assert.equal(
        await page.locator("#model-table tbody tr").count(),
        workload.rows.length,
      );
      await page.selectOption("#model-metric", "bi");
      await page.selectOption("#model-metric", "qnbi");
    }
    await page.selectOption("#workload", "sharegpt");
    await page.selectOption("#model-family", "qwen-3");
    assert.ok((await page.locator("#model-table tbody tr").count()) < 21);
    await page.selectOption("#model-family", "gpt-oss");
    await page.selectOption("#workload", "crosscodeeval");
    assert.equal(await page.locator("#model-family").inputValue(), "all");
    await page.uncheck("#log-scale");
    await page.selectOption("#reasoning-metric", "quality");
    await page.selectOption("#gpu-family", "gemma");
    assert.equal(await page.locator("#gpu-figures").innerText(), "FIGURE 27");
    await page.selectOption("#gpu-metric", "bi");
    await page.locator("#copy-citation").click();
    await expect(page.locator("#copy-status")).toHaveText(/Citation (copied|selected)/);
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `${name}: figures, assets, interactions, data tables, and layout passed`,
    );
  }
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto("http://127.0.0.1:4179/BIRDS/");
  assert.ok(await page.locator("noscript").isVisible());
  await context.close();
  console.log("Processed data validation and no-JavaScript fallback passed");
} finally {
  await browser?.close();
  server.kill();
}
