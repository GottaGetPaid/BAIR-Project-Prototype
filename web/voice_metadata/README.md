# Voice Metadata Collection

This folder contains detailed voice input metadata collected from Deepgram, **without storing any audio files**.

## What is Collected

Each voice input generates a JSON file containing:

### 1. **Word-Level Timestamps**
Every spoken word with:
- `word`: The raw word
- `punctuated_word`: Word with proper punctuation
- `start`: Start time in seconds
- `end`: End time in seconds  
- `confidence`: Confidence score (0-1)

### 2. **Pause Detection**
Automatically detected pauses between words:
- `after_word` / `before_word`: Context for the pause
- `duration_seconds`: Length of the pause
- `timestamp`: When the pause occurred
- Categorized as:
  - **Short pause** (>0.3s): single period `.`
  - **Medium pause** (>0.5s): double period `..`
  - **Long pause** (>1.0s): ellipsis `...`

### 3. **Descriptive Transcript**
A text transcript with semantic boundaries marked:
```
Hello. How are you.. What is the square root of 65... I'm waiting.
```

### 4. **Session Metadata**
- `session_start`: ISO timestamp when recording started
- `total_duration_seconds`: Total speech duration
- `word_count`: Number of words spoken
- `utterance_count`: Number of natural speech segments
- `saved_at`: When the metadata was saved

### 5. **Utterance Boundaries**
Natural speech segments detected by Deepgram's AI:
- Each utterance marks a complete thought or sentence
- Includes timestamp of utterance end

## Example Output

```json
{
  "transcript": "What is the square root of 65?",
  "session_start": "2025-10-23T20:15:30.123Z",
  "total_duration_seconds": 2.45,
  "word_count": 7,
  "utterance_count": 1,
  "words": [
    {
      "word": "what",
      "punctuated_word": "What",
      "start": 0.0,
      "end": 0.24,
      "confidence": 0.99
    },
    {
      "word": "is",
      "punctuated_word": "is",
      "start": 0.24,
      "end": 0.36,
      "confidence": 0.98
    }
    // ... more words
  ],
  "pauses": [
    {
      "after_word": "square",
      "before_word": "root",
      "duration_seconds": 0.45,
      "timestamp": 1.2
    }
  ],
  "descriptive_transcript": "What is the square.. root of 65?",
  "utterances": [
    {
      "type": "utterance_end",
      "timestamp": "2025-10-23T20:15:32.567Z"
    }
  ],
  "saved_at": "2025-10-23T20:15:33.000Z"
}
```

## Use Cases

This metadata enables detailed analysis:

1. **Speech Pattern Analysis**: Study pause patterns, speaking rate, hesitations
2. **Confidence Tracking**: Monitor recognition confidence per word
3. **Temporal Analysis**: Analyze when users pause, speed up, or slow down
4. **UX Research**: Understand how users interact with voice interfaces
5. **Training Data**: Generate labeled datasets without storing audio

## Privacy

✅ **No audio files are stored**  
✅ Only text transcripts and timing metadata are saved  
✅ Suitable for privacy-conscious applications
