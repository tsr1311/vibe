#!/usr/bin/env python3
"""
Faster-whisper transcription script for Vibe
"""
import sys
import json
import os
from faster_whisper import WhisperModel

def main():
    if len(sys.argv) != 3:
        print("Usage: python transcribe_faster_whisper.py <model> <audio_file>", file=sys.stderr)
        sys.exit(1)

    model_name = sys.argv[1]
    audio_file = sys.argv[2]

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Initialize the model
        model = WhisperModel(model_name, device="auto", compute_type="default")

        # Transcribe
        segments, info = model.transcribe(audio_file, beam_size=5)

        # Convert segments to the expected format
        segment_list = []
        for segment in segments:
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        # Output JSON
        output = {
            "segments": segment_list,
            "language": info.language,
            "language_probability": info.language_probability
        }

        print(json.dumps(output))

    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
