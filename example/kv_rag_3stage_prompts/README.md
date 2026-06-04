# KV-RAG Three-Stage Validation Prompts

This directory is formatted for `MultiTextConcatDataset` directory mode:

```text
example/kv_rag_3stage_prompts/
  caption/
    000_man_turnaround/
      0.json
      1.json
      2.json
      shot_durations.txt
```

Each subfolder is one validation video. The three JSON files are stage captions, and `shot_durations.txt` assigns the number of latent chunks to each stage. With the default `num_output_frames: 144` and `num_frame_per_block: 8`, each sample has 18 chunks, split as `6 6 6`.

The active examples are human-only. They intentionally repeat the same appearance anchors in every stage. That makes failures easy to see: faces, clothing, accessories, props, and distinctive marks should remain stable when the subject turns away and returns.
