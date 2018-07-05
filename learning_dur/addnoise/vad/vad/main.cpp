//
//  main.cpp
//  SSPTest
//
//  Created by GaoPeng on 15/5/1.
//  Copyright (c) 2015年 Rokid. All rights reserved.
//

#include <string.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "NNVadIntf.h"

typedef  struct WAVE_HEADERTag
{
    char RIFF[4];
    int Whgilelen;
    char WAVEfmt[8];
    int version;
    short  FormatTag;
    short  Channels;
    int SamplePerSec;
    int AvgBytesPerSec;
    short  blockalign;
    short  BitPerSample;
    //char EXTEND[8];
    char data[4];
    int Pcmfilelen;
} WAVE_HEADER;


#ifndef PI
#define PI	3.14159265358979323
#endif

void FillWaveHeader(WAVE_HEADER *pHdr, int nChunkLen, int nChannels, int nSampleFreq, int nBytesPerSample) {
    WAVE_HEADER &waveHeader = *pHdr;
    memset(&waveHeader, 0, sizeof(waveHeader));
    waveHeader.RIFF[0]        = 'R';
    waveHeader.RIFF[1]        = 'I';
    waveHeader.RIFF[2]        = 'F';
    waveHeader.RIFF[3]        = 'F';
    waveHeader.Whgilelen      = sizeof(waveHeader) + nChunkLen; //chunk length
    waveHeader.WAVEfmt[0]     = 'W';
    waveHeader.WAVEfmt[1]     = 'A';
    waveHeader.WAVEfmt[2]     = 'V';
    waveHeader.WAVEfmt[3]     = 'E';
    waveHeader.WAVEfmt[4]     = 'f';
    waveHeader.WAVEfmt[5]     = 'm';
    waveHeader.WAVEfmt[6]     = 't';
    waveHeader.WAVEfmt[7]     = ' ';
    waveHeader.version        = 16;    //chunk1 length
    waveHeader.FormatTag      = 0x0001;
    waveHeader.Channels       = nChannels;
    waveHeader.SamplePerSec   = nSampleFreq;
    waveHeader.AvgBytesPerSec = nSampleFreq*nBytesPerSample*nChannels;
    waveHeader.blockalign     = nBytesPerSample * nChannels;
    waveHeader.BitPerSample   = nBytesPerSample*8;
    waveHeader.data[0]        = 'd';
    waveHeader.data[1]        = 'a';
    waveHeader.data[2]        = 't';
    waveHeader.data[3]        = 'a';
    waveHeader.Pcmfilelen     = nChunkLen; //chunk length
}

#define INFORMAT_INT16BIT   0
#define INFORMAT_INT24BIT   1
#define INFORMAT_FLOAT      2
#define OUTFORMAT_INT16BIT  0
#define OUTFORMAT_FLOAT     1

int ReadWaveFile(const char *szFileName, int iInFormat, int iOutFormat,
                 void **pOutData, int *piOutLen, float scale=1.0f) {
    if (!szFileName || !pOutData)
        return -1;
    FILE *fpFile = fopen(szFileName, "rb");
    if (!fpFile)
        return -1;
    fseek(fpFile, 0, SEEK_END);
    int iSize = (int) ftell(fpFile);
    int iSkip = 0;
    if (szFileName + strlen(szFileName) - 4 == strstr(szFileName, ".wav")) {
        iSkip = 44;
        iSize -= iSkip;
    }
    fseek(fpFile, iSkip, SEEK_SET);

    void *&pData = *pOutData;
    if (iOutFormat == OUTFORMAT_INT16BIT)
        pData = new short[iSize];
    else if (iOutFormat == OUTFORMAT_FLOAT)
        pData = new float[iSize];

    char *pDataPtr = (char*) pData;
    int iReadSize = 2;
    if (iInFormat == INFORMAT_INT16BIT)
        iReadSize = 2;
    else if (iInFormat == INFORMAT_INT24BIT)
        iReadSize = 3;
    else if (iInFormat == INFORMAT_FLOAT)
        iReadSize = 4;
    int iReadNum = 0;
    while (!feof(fpFile)) {
        int iReadBuf = 0;
        size_t iRead = fread(&iReadBuf, iReadSize, 1, fpFile);
        if (iRead != 1)
            break;
        if (iInFormat == INFORMAT_INT16BIT && iReadBuf >= 0x8000)
            iReadBuf |= 0xffff0000;
        else if (iInFormat == INFORMAT_INT24BIT && iReadBuf >= 0x800000)
            iReadBuf |= 0xff000000;
        if (iOutFormat == OUTFORMAT_INT16BIT) {
            if (iInFormat == INFORMAT_FLOAT)
                *(short*) pDataPtr = *(float*)&iReadBuf * scale;
            else
                *(short*) pDataPtr = (short) (iReadBuf * scale);
            pDataPtr += sizeof(short);
        } else if (iOutFormat == OUTFORMAT_FLOAT) {
            if (iInFormat == INFORMAT_FLOAT)
                *(float*) pDataPtr = *(float*)&iReadBuf;
            else
                *(float*) pDataPtr = (float) iReadBuf;
            *(float*) pDataPtr *= scale;
            pDataPtr += sizeof(float);
        }
        iReadNum++;
    }
    if (piOutLen)
        *piOutLen = iReadNum;
    
    fclose(fpFile);
    return 0;
}

std::string& std_string_trim(std::string &s) {
    if (s.empty()) {
        return s;
    }
    
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
    return s;
}

void create_full_path(const char* pPath) {
    char tmpPath[260];
    memset(tmpPath, 0, sizeof(tmpPath));
    for (size_t i = 0; i < sizeof(tmpPath); i++) {
        if ((pPath[i]) == '/') {
            mkdir(tmpPath, 0777);
        } else if (!(pPath[i]))
            break;
        tmpPath[i] = pPath[i];
    }
    //mkdir(tmpPath, 0777);
}

int TestVad() {
    std::string strDir = "/Users/gaopeng/Downloads/";
    // mic files
    const char *files[] = {
        "vad_waves.pcm",
        //"13701237076.wav",
    };
    
    for (int i=0; i < sizeof(files)/sizeof(void*); i++) {
        float *dataMic;
        int iLenMic;
        std::string strFile = strDir;
        strFile += files[i];
        //printf("%s\n", strFile.c_str());
        float fScale = 32767.0;
        //ReadWaveFile(strFile.c_str(), INFORMAT_INT16BIT, OUTFORMAT_FLOAT,
        //             (void**) &dataMic, &iLenMic);
        ReadWaveFile(strFile.c_str(), INFORMAT_FLOAT, OUTFORMAT_FLOAT,
                     (void**) &dataMic, &iLenMic, fScale);

        int sampleRate = 16000;
        int frameSizeMs = 10;
        int frameSize = sampleRate * frameSizeMs / 1000;
        int frameNum = iLenMic / frameSize;

        std::vector<std::pair<int, int> > vadResults;
        std::vector<float*> vadWaves;
        std::vector<int> vadWaveLengths;
        VD_HANDLE hVad = VD_NewVad(1);
        int val = 60;
        //VD_SetVadParam(hVad, VD_PARAM_MINSILFRAMENUM, &nMinSilFrameNum);
        //nMinSilFrameNum = 20;
        //VD_SetVadParam(hVad, VD_PARAM_MINVOCFRAMENUM, &nMinSilFrameNum);
        val = 300;
        VD_SetVadParam(hVad, VD_PARAM_MINENERGY, &val);
        val = 1;
        VD_SetVadParam(hVad, VD_PARAM_ENBLEPITCH, &val);
        val = 0;
        VD_SetVadParam(hVad, VD_PARAM_MINVOCFRAMENUM, &val);
        val = 50;
        VD_SetVadParam(hVad, VD_PARAM_MINSILFRAMENUM, &val);
        int prevOffset = 0;
        for (int i = 0; i < frameNum; i++) {
            //printf("frame: %d", i);
            int bIsEnd = 0;
            if (i == frameNum-1)
                bIsEnd = 1;
            int nIsAec = 0;

            VD_InputFloatWave(hVad, dataMic+i*frameSize, frameSize, bIsEnd, nIsAec);
            int startFrame = VD_GetVoiceStartFrame(hVad);
            if (startFrame >= 0) {
                //printf("\tstart: %d", startFrame + prevOffset);
                int stopFrame = VD_GetVoiceStopFrame(hVad);
                if (stopFrame > 0) {
                    //prevOffset = VD_GetOffsetFrame(hVad);
                    //printf("\tstop: %d", stopFrame + prevOffset);
                    //printf("frame: %d\tstart: %d\tstop: %d\n", i, startFrame+prevOffset, stopFrame + prevOffset);
                    vadResults.push_back(std::pair<int,int>(startFrame+prevOffset, stopFrame + prevOffset));

                    int nWaveLength = frameSize*VD_GetVoiceFrameNum(hVad);
                    float *pWaves = new float[nWaveLength];
                    memcpy(pWaves, VD_GetFloatVoice(hVad), nWaveLength*sizeof(float));
                    vadWaves.push_back(pWaves);
                    vadWaveLengths.push_back(nWaveLength);

                    VD_RestartVad(hVad);
                    prevOffset = i+1;
                }
            }
            //printf("\n");
        }
        VD_DelVad(hVad);

        for (int i=0; i<iLenMic; i++) {
            dataMic[i] /= 0x7fff;
        }
        std::string strFileOut = strDir+files[i];
        strFileOut += ".vad.wav";
        FILE *fpOut = fopen(strFileOut.c_str(), "wb");

        WAVE_HEADER waveHeader;
        FillWaveHeader(&waveHeader, frameNum*frameSize*2, 1, sampleRate, 2);
        fwrite(&waveHeader, 1, sizeof(waveHeader), fpOut);

        for (int i = 0; i < frameNum; i++) {
            for (int j=0; j<vadResults.size(); j++) {
                std::pair<int,int> &se = vadResults[j];
                if (se.first == i) {
                    dataMic[i*frameSize] = 1.0f;
                    //fwrite(&f, 1, sizeof(float), fpOut);
                }
                else if (se.second == i) {
                    dataMic[i*frameSize+frameSize-1] = -1.0f;
                    //fwrite(&f, 1, sizeof(float), fpOut);
                }
            }
            //fwrite(dataMic+i*frameSize, frameSize, sizeof(float), fpOut);
            for (int j=0; j<frameSize; j++) {
                short sample = dataMic[i*frameSize+j] * 0x7fff;
                if (dataMic[i*frameSize+j] == 1.0f)
                    sample = 32767;
                if (dataMic[i*frameSize+j] == -1.0f)
                    sample = -32768;
                fwrite(&sample, 1, 2, fpOut);
            }
        }
        fclose(fpOut);

        strFileOut = strDir+files[i];
        strFileOut += ".vadwave.pcm";
        fpOut = fopen(strFileOut.c_str(), "wb");
        for (int i = 0; i < vadWaves.size(); i++) {
            fwrite(vadWaves[i], vadWaveLengths[i], sizeof(float), fpOut);
            delete[] vadWaves[i];
        }
        fclose(fpOut);

        delete[] dataMic;
    }
    return 0;
}

int TestVadShort() {
    std::string strDir = "/Users/gaopeng/Downloads/";
    // mic files
    const char *files[] = {
        "rqxtsj.wav",
    };

    for (int i=0; i < sizeof(files)/sizeof(void*); i++) {
        short *dataMic;
        int iLenMic;
        std::string strFile = strDir;
        strFile += files[i];
        //printf("%s\n", strFile.c_str());
        float fScale = 1.0f;
        //ReadWaveFile(strFile.c_str(), INFORMAT_INT16BIT, OUTFORMAT_FLOAT,
        //             (void**) &dataMic, &iLenMic);
        ReadWaveFile(strFile.c_str(), INFORMAT_INT16BIT, INFORMAT_INT16BIT,
                     (void**)&dataMic, &iLenMic, fScale);

        int sampleRate = 16000;
        int frameSizeMs = 10;
        int frameSize = sampleRate * frameSizeMs / 1000;
        int frameNum = iLenMic / frameSize;

        std::vector<std::pair<int, int> > vadResults;
        std::vector<short*> vadWaves;
        std::vector<int> vadWaveLengths;
        VD_HANDLE hVad = VD_NewVad(1);
        int val = 60;
        //VD_SetVadParam(hVad, VD_PARAM_MINSILFRAMENUM, &nMinSilFrameNum);
        //nMinSilFrameNum = 20;
        //VD_SetVadParam(hVad, VD_PARAM_MINVOCFRAMENUM, &nMinSilFrameNum);
        val = 300;
        VD_SetVadParam(hVad, VD_PARAM_MINENERGY, &val);
        val = 1;
        VD_SetVadParam(hVad, VD_PARAM_ENBLEPITCH, &val);
        val = 50;
        VD_SetVadParam(hVad, VD_PARAM_MINSILFRAMENUM, &val);
        int prevOffset = 0;
        for (int i = 0; i < frameNum; i++) {
            //printf("frame: %d", i);
            int bIsEnd = 0;
            if (i == frameNum-1)
                bIsEnd = 1;
            int nIsAec = 0;

            //if (i == 120)
            //    VD_ResetVad(hVad);

            VD_InputWave(hVad, dataMic+i*frameSize, frameSize, bIsEnd, nIsAec);
            int startFrame = VD_GetVoiceStartFrame(hVad);
            if (startFrame >= 0) {
                //printf("\tstart: %d", startFrame + prevOffset);
                int stopFrame = VD_GetVoiceStopFrame(hVad);
                if (stopFrame > 0) {
                    //prevOffset = VD_GetOffsetFrame(hVad);
                    //printf("\tstop: %d", stopFrame + prevOffset);
                    //printf("frame: %d\tstart: %d\tstop: %d\n", i, startFrame+prevOffset, stopFrame + prevOffset);
                    vadResults.push_back(std::pair<int,int>(startFrame+prevOffset, stopFrame + prevOffset));

                    int nWaveLength = frameSize*VD_GetVoiceFrameNum(hVad);
                    short *pWaves = new short[nWaveLength];
                    memcpy(pWaves, VD_GetVoice(hVad), nWaveLength*sizeof(short));
                    vadWaves.push_back(pWaves);
                    vadWaveLengths.push_back(nWaveLength);

                    VD_RestartVad(hVad);
                    prevOffset = i+1;
                }
            }
            //printf("\n");
        }
        VD_DelVad(hVad);

        std::string strFileOut = strDir+files[i];
        strFileOut += ".vad.wav";
        FILE *fpOut = fopen(strFileOut.c_str(), "wb");

        WAVE_HEADER waveHeader;
        FillWaveHeader(&waveHeader, frameNum*frameSize*2, 1, sampleRate, 2);
        fwrite(&waveHeader, 1, sizeof(waveHeader), fpOut);

        for (int i = 0; i < frameNum; i++) {
            for (int j=0; j<vadResults.size(); j++) {
                std::pair<int,int> &se = vadResults[j];
                if (se.first == i) {
                    dataMic[i*frameSize] = 32767;
                    //fwrite(&f, 1, sizeof(float), fpOut);
                }
                else if (se.second == i) {
                    dataMic[i*frameSize+frameSize-1] = -32768;
                    //fwrite(&f, 1, sizeof(float), fpOut);
                }
            }
            fwrite(dataMic+i*frameSize, frameSize, sizeof(short), fpOut);
        }
        fclose(fpOut);

        strFileOut = strDir+files[i];
        strFileOut += ".vadwave.pcm";
        fpOut = fopen(strFileOut.c_str(), "wb");
        for (int i = 0; i < vadWaves.size(); i++) {
            fwrite(vadWaves[i], vadWaveLengths[i], sizeof(short), fpOut);
            delete[] vadWaves[i];
        }
        fclose(fpOut);
        
        delete[] dataMic;
    }
    return 0;
}

int VadFile(const char *szInputFile, int inFormat, int inSampleRate,
            float inScale, int vadMode, const char *szOutFile, int outFormat, float outScale) {
    if (inSampleRate != 16000)
        return -1;
    float *fileData = 0;
    int iFileLen = 0;
    int nRet = ReadWaveFile(szInputFile, inFormat, OUTFORMAT_FLOAT,
                            (void**) &fileData, &iFileLen, inScale);
    if (nRet != 0)
        return nRet;

    int frameSizeMs = 10;
    int frameSize = inSampleRate * frameSizeMs / 1000;
    int frameNum = iFileLen / frameSize;

    std::vector<std::pair<int, int> > vadResults;
    int maxVadIdx = -1, maxVadLen = 0;
    VD_HANDLE hVad = VD_NewVad(vadMode);
    int prevOffset = 0;
    for (int i = 0; i < frameNum; i++) {
        int bIsEnd = 0;
        if (i == frameNum-1)
            bIsEnd = 1;
        int nIsAec = 0;
        VD_InputFloatWave(hVad, fileData+i*frameSize, frameSize, bIsEnd, nIsAec);
        int startFrame = VD_GetVoiceStartFrame(hVad);
        if (startFrame >= 0) {
            int stopFrame = VD_GetVoiceStopFrame(hVad);
            if (stopFrame > 0) {
                int vadLen = stopFrame - startFrame + 1;
                if (maxVadIdx == -1) {
                    maxVadIdx = 0;
                    maxVadLen = vadLen;
                }
                else if (vadLen > maxVadLen) {
                    maxVadIdx = (int)vadResults.size();
                    maxVadLen = vadLen;
                }
                vadResults.push_back(std::pair<int,int>(startFrame+prevOffset, stopFrame + prevOffset));
                VD_RestartVad(hVad);
                prevOffset = i+1;
            }
        }
    }
    VD_DelVad(hVad);
    if (maxVadIdx == -1)
        return -1;

    create_full_path(szOutFile);
    FILE *fpOut = fopen(szOutFile, "wb");
    if (fpOut) {
        int startFrame = vadResults[maxVadIdx].first;
        int stopFrame = vadResults[maxVadIdx].second;
        for (int i = startFrame * frameSize; i <= stopFrame * frameSize; i++) {
            if (outFormat == OUTFORMAT_FLOAT) {
                float iSample = (float) (fileData[i] * outScale);
                fwrite(&iSample, sizeof(iSample), 1, fpOut);
            } else if (outFormat == OUTFORMAT_INT16BIT) {
                short iSample = (short) (fileData[i] * outScale);
                fwrite(&iSample, sizeof(iSample), 1, fpOut);
            }
        }
        fclose(fpOut);
    }
    delete[] fileData;

    return 0;
}

int VadFileMain(int argc, const char * argv[]) {
    if (argc < 8) {
        printf("Usage:\nSSPVad input_file in_format in_sample_rate in_scale vad_mode output_file out_format out_scale\n");
        printf("\tin_format: 0(int16), 1(int24), 2(float)\n\tout_format: 0(int16), 1(float)\n");
        printf("\tvad_mode: 0(dnn), 1(energy), 2(dnn_energy)\n");
        printf("\tin_sample_rate only support 16000\n");
        return -1;
    }
    const char *szInputFile = argv[1];
    int inFormat = atoi(argv[2]);
    int inSampleRate = atoi(argv[3]);
    float inScale = atof(argv[4]);
    int vadMode = atoi(argv[5]);
    const char *szOutFile = argv[6];
    int outFormat = atoi(argv[7]);
    float outScale = atof(argv[8]);
    return VadFile(szInputFile, inFormat, inSampleRate, inScale, vadMode, szOutFile,
                   outFormat, outScale);
}

int main(int argc, const char * argv[]) {
    TestVad();
    //TestVadShort();

    //return VadFileMain(argc, argv);

    return 0;
}
