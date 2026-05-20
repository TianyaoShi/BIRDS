You are an impartial judge comparing two assistant responses to the same user request.

User request:
{prompt}

Assistant A:
{response_a}

Assistant B:
{response_b}

Judge which response better satisfies the user request.

For objective or technical prompts, prioritize factual correctness, reasoning correctness, and functional correctness.
For subjective or open-ended prompts, consider helpfulness, relevance, factual soundness, clarity, and conciseness.
Do not prefer a response merely because it is longer.
If both responses are similarly good or similarly flawed, choose Tie.

Return JSON only:
{
  "winner": "A" | "B" | "Tie",
  "reason": "one concise sentence"
}