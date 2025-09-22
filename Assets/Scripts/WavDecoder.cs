using System;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Application = UnityEngine.Application;

// Robust WAV decoder for PCM 16/24/32-bit and IEEE float32. Returns interleaved float samples.
public static class WavDecoder
{
    // Common format codes
    const ushort WAVE_FORMAT_PCM = 0x0001;
    const ushort WAVE_FORMAT_IEEE_FLOAT = 0x0003;
    const ushort WAVE_FORMAT_EXTENSIBLE = 0xFFFE;

    public static bool TryDecode(byte[] wavBytes, out float[] samples, out int channels, out int sampleRate)
    {
        samples = Array.Empty<float>();
        channels = 0;
        sampleRate = 0;

        try
        {
            if (wavBytes == null || wavBytes.Length < 12) return false;

            // Check RIFF/WAVE
            if (Encoding.ASCII.GetString(wavBytes, 0, 4) != "RIFF" ||
                Encoding.ASCII.GetString(wavBytes, 8, 4) != "WAVE")
            {
                Debug.LogError("[WavDecoder] Not a RIFF/WAVE file.");
                return false;
            }

            long pos = 12; // after RIFF header
            bool foundFmt = false;
            bool foundData = false;

            ushort audioFormat = WAVE_FORMAT_PCM;
            ushort bitsPerSample = 16;
            uint dataSize = 0;
            int dataOffset = -1;

            // fmt fields
            int fmtDataPos = -1;

            while (pos + 8 <= wavBytes.Length)
            {
                // Read chunk header
                string chunkId = Encoding.ASCII.GetString(wavBytes, (int)pos, 4);
                uint chunkSize = BitConverter.ToUInt32(wavBytes, (int)pos + 4);
                long chunkDataPos = pos + 8;

                // Basic sanity; stop if header says something impossible
                if (chunkDataPos < 0 || chunkDataPos > wavBytes.Length)
                {
                    Debug.LogError("[WavDecoder] Corrupt chunk header.");
                    return false;
                }

                if (chunkId == "fmt ")
                {
                    // We only need standard fields
                    if (chunkDataPos + 16 > wavBytes.Length) return false;
                    audioFormat = BitConverter.ToUInt16(wavBytes, (int)chunkDataPos + 0);
                    channels = BitConverter.ToUInt16(wavBytes, (int)chunkDataPos + 2);
                    sampleRate = BitConverter.ToInt32(wavBytes, (int)chunkDataPos + 4);
                    // byteRate    = BitConverter.ToInt32 (wavBytes, (int)chunkDataPos + 8);
                    // blockAlign  = BitConverter.ToUInt16(wavBytes, (int)chunkDataPos + 12);
                    bitsPerSample = BitConverter.ToUInt16(wavBytes, (int)chunkDataPos + 14);

                    fmtDataPos = (int)chunkDataPos;
                    foundFmt = true;
                }
                else if (chunkId == "data")
                {
                    // Ensure we don't exceed the file; clamp if needed
                    long maxData = wavBytes.Length - chunkDataPos;
                    if ((long)chunkSize > maxData)
                    {
                        Debug.LogWarning("[WavDecoder] Data chunk size exceeds file length; clamping.");
                        chunkSize = (uint)Mathf.Max(0, (int)maxData);
                    }

                    dataSize = chunkSize;
                    dataOffset = (int)chunkDataPos;
                    foundData = true;
                }
                // Skip other chunks like 'JUNK', 'LIST', 'fact', 'cue ', 'smpl', etc.

                // Advance to next chunk (with alignment/padding)
                pos = chunkDataPos + chunkSize;
                if ((pos & 1) == 1) pos++;

                if (foundFmt && foundData) break;
            }

            if (!foundFmt || !foundData || dataOffset < 0 || channels <= 0 || sampleRate <= 0)
            {
                Debug.LogError("[WavDecoder] Missing fmt/data chunk or invalid header.");
                return false;
            }

            // Some providers use WAVE_FORMAT_EXTENSIBLE; infer PCM vs float from bitsPerSample
            ushort effectiveFormat = audioFormat;
            if (audioFormat == WAVE_FORMAT_EXTENSIBLE)
            {
                // Without parsing the GUID, inference works well:
                // - 32 bits -> float
                // - else (16/24) -> PCM int
                effectiveFormat = (bitsPerSample == 32) ? WAVE_FORMAT_IEEE_FLOAT : WAVE_FORMAT_PCM;
            }

            int bytesPerSample = bitsPerSample / 8;
            if (bytesPerSample <= 0)
            {
                Debug.LogError($"[WavDecoder] Invalid bitsPerSample: {bitsPerSample}");
                return false;
            }

            long bytesPerFrame = (long)bytesPerSample * (long)channels;
            if (bytesPerFrame <= 0)
            {
                Debug.LogError("[WavDecoder] Invalid bytesPerFrame.");
                return false;
            }

            // Clamp dataSize to remaining file length if necessary
            long maxRemain = wavBytes.Length - dataOffset;
            if (dataSize == 0 || dataSize > maxRemain)
                dataSize = (uint)Mathf.Max(0, (int)maxRemain);

            long totalFrames = (long)dataSize / bytesPerFrame;
            if (totalFrames <= 0)
            {
                Debug.LogError("[WavDecoder] No audio frames found.");
                return false;
            }

            long totalSamplesLong = totalFrames * channels;
            if (totalSamplesLong > int.MaxValue)
            {
                Debug.LogError("[WavDecoder] Unreasonably large sample count.");
                return false;
            }

            int totalSamples = (int)totalSamplesLong;
            samples = new float[totalSamples];

            // Decode
            switch (effectiveFormat)
            {
                case WAVE_FORMAT_PCM:
                    {
                        switch (bitsPerSample)
                        {
                            case 16:
                                {
                                    int count = (int)(totalFrames * channels);
                                    for (int i = 0; i < count; i++)
                                    {
                                        int offset = dataOffset + i * 2;
                                        if (offset + 2 > wavBytes.Length) { samples = Array.Empty<float>(); return false; }
                                        short s = BitConverter.ToInt16(wavBytes, offset);
                                        samples[i] = Mathf.Clamp(s / 32768f, -1f, 1f);
                                    }
                                    break;
                                }
                            case 24:
                                {
                                    int count = (int)(totalFrames * channels);
                                    for (int i = 0; i < count; i++)
                                    {
                                        int b = dataOffset + i * 3;
                                        if (b + 3 > wavBytes.Length) { samples = Array.Empty<float>(); return false; }
                                        int val = wavBytes[b] | (wavBytes[b + 1] << 8) | (wavBytes[b + 2] << 16);
                                        if ((val & 0x800000) != 0) val |= unchecked((int)0xFF000000); // sign-extend
                                        samples[i] = Mathf.Clamp(val / 8388608f, -1f, 1f);
                                    }
                                    break;
                                }
                            case 32:
                                {
                                    // 32-bit PCM integer
                                    int count = (int)(totalFrames * channels);
                                    for (int i = 0; i < count; i++)
                                    {
                                        int offset = dataOffset + i * 4;
                                        if (offset + 4 > wavBytes.Length) { samples = Array.Empty<float>(); return false; }
                                        int ival = BitConverter.ToInt32(wavBytes, offset);
                                        samples[i] = Mathf.Clamp(ival / 2147483648f, -1f, 1f);
                                    }
                                    break;
                                }
                            default:
                                Debug.LogError($"[WavDecoder] Unsupported PCM bit depth: {bitsPerSample}");
                                return false;
                        }
                        break;
                    }
                case WAVE_FORMAT_IEEE_FLOAT:
                    {
                        if (bitsPerSample != 32)
                        {
                            Debug.LogError($"[WavDecoder] Float WAV with bitsPerSample={bitsPerSample} not supported.");
                            return false;
                        }
                        int count = (int)(totalFrames * channels);
                        for (int i = 0; i < count; i++)
                        {
                            int offset = dataOffset + i * 4;
                            if (offset + 4 > wavBytes.Length) { samples = Array.Empty<float>(); return false; }
                            float f = BitConverter.ToSingle(wavBytes, offset);
                            samples[i] = Mathf.Clamp(f, -1f, 1f);
                        }
                        break;
                    }
                default:
                    Debug.LogError($"[WavDecoder] Unsupported audio format code: 0x{effectiveFormat:X4}");
                    return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            Debug.LogError("[WavDecoder] Exception while decoding: " + ex.Message);
            samples = Array.Empty<float>();
            channels = 0;
            sampleRate = 0;
            return false;
        }
    }
}