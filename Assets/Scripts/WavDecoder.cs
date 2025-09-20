using System;
using System.Diagnostics;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Application  = UnityEngine.Application;

// Minimal WAV decoder for PCM 16/24-bit and IEEE float32. Returns interleaved float samples.
public static class WavDecoder
{
    public static bool TryDecode(byte[] wavBytes, out float[] samples, out int channels, out int sampleRate)
    {
        samples = Array.Empty<float>();
        channels = 0;
        sampleRate = 0;

        if (wavBytes == null || wavBytes.Length < 44) return false;

        // Check RIFF/WAVE
        if (Encoding.ASCII.GetString(wavBytes, 0, 4) != "RIFF" ||
            Encoding.ASCII.GetString(wavBytes, 8, 4) != "WAVE")
        {
            Debug.LogError("[WavDecoder] Not a RIFF/WAVE file.");
            return false;
        }

        int pos = 12; // start after RIFF header
        ushort audioFormat = 1;
        ushort bitsPerSample = 16;
        uint dataSize = 0;
        int dataOffset = -1;

        while (pos + 8 <= wavBytes.Length)
        {
            string chunkId = Encoding.ASCII.GetString(wavBytes, pos, 4);
            int chunkSize = BitConverter.ToInt32(wavBytes, pos + 4);
            int chunkDataPos = pos + 8;

            if (chunkId == "fmt ")
            {
                if (chunkDataPos + 16 > wavBytes.Length) return false;
                audioFormat = BitConverter.ToUInt16(wavBytes, chunkDataPos + 0);  // 1=PCM, 3=IEEE float
                channels = BitConverter.ToUInt16(wavBytes, chunkDataPos + 2);
                sampleRate = BitConverter.ToInt32(wavBytes, chunkDataPos + 4);
                // byteRate    = BitConverter.ToInt32 (wavBytes, chunkDataPos + 8);
                // blockAlign  = BitConverter.ToUInt16(wavBytes, chunkDataPos + 12);
                bitsPerSample = BitConverter.ToUInt16(wavBytes, chunkDataPos + 14);
            }
            else if (chunkId == "data")
            {
                dataSize = (uint)chunkSize;
                dataOffset = chunkDataPos;
                break;
            }

            pos = chunkDataPos + chunkSize;
            if (pos % 2 == 1) pos++; // alignment
        }

        if (dataOffset < 0 || channels <= 0 || sampleRate <= 0)
        {
            Debug.LogError("[WavDecoder] Missing fmt/data chunk or invalid header.");
            return false;
        }

        int bytesPerSample = bitsPerSample / 8;
        int totalSamples = (int)(dataSize / bytesPerSample);
        if (totalSamples <= 0) return false;

        samples = new float[totalSamples];
        int idx = 0;

        try
        {
            if (audioFormat == 1) // PCM integer
            {
                switch (bitsPerSample)
                {
                    case 16:
                        for (int i = 0; i < totalSamples; i++)
                        {
                            short s = BitConverter.ToInt16(wavBytes, dataOffset + i * 2);
                            samples[idx++] = Mathf.Clamp(s / 32768f, -1f, 1f);
                        }
                        break;

                    case 24:
                        for (int i = 0; i < totalSamples; i++)
                        {
                            int b = dataOffset + i * 3;
                            int val = wavBytes[b] | (wavBytes[b + 1] << 8) | (wavBytes[b + 2] << 16);
                            // sign-extend 24-bit
                            if ((val & 0x800000) != 0) val |= unchecked((int)0xFF000000);
                            samples[idx++] = Mathf.Clamp(val / 8388608f, -1f, 1f);
                        }
                        break;

                    case 32:
                        for (int i = 0; i < totalSamples; i++)
                        {
                            int ival = BitConverter.ToInt32(wavBytes, dataOffset + i * 4);
                            samples[idx++] = Mathf.Clamp(ival / 2147483648f, -1f, 1f);
                        }
                        break;

                    default:
                        Debug.LogError($"[WavDecoder] Unsupported PCM bit depth: {bitsPerSample}");
                        return false;
                }
            }
            else if (audioFormat == 3) // IEEE float32
            {
                if (bitsPerSample != 32)
                {
                    Debug.LogError($"[WavDecoder] Float WAV with bitsPerSample={bitsPerSample} not supported.");
                    return false;
                }
                for (int i = 0; i < totalSamples; i++)
                {
                    float f = BitConverter.ToSingle(wavBytes, dataOffset + i * 4);
                    samples[idx++] = Mathf.Clamp(f, -1f, 1f);
                }
            }
            else
            {
                Debug.LogError($"[WavDecoder] Unsupported audio format code: {audioFormat}");
                return false;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("[WavDecoder] Exception while decoding: " + e.Message);
            return false;
        }

        return true;
    }
}