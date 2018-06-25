//
//  main.cpp
//  SSPTest
//
//  Created by lcfan on 2018/5/22.
//  Copyright (c) 2018年 Rokid. All rights reserved.
//

#include <string.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "NNVadIntf.h"

#ifndef PI
#define PI	3.14159265358979323
#endif

#define SKIP_WAVE_HEAD 44
#define MAX_WAV_LENGTH 960000  //60s * 16000Hz
#define sampleRate 16000
#define frameSizeMs 10

int TestVadShort(const short* dataMic, const int iLenMic, std::vector<int> &vads) {
    int frameSize = sampleRate * frameSizeMs / 1000;
    int frameNum = iLenMic / frameSize;

    std::vector<std::pair<int, int> > vadResults;
    VD_HANDLE hVad = VD_NewVad(1);
    int val = 300;
    VD_SetVadParam(hVad, VD_PARAM_MINENERGY, &val);
    val = 1;
    VD_SetVadParam(hVad, VD_PARAM_ENBLEPITCH, &val);
    val = 5;
    VD_SetVadParam(hVad, VD_PARAM_MINSILFRAMENUM, &val);
    int prevOffset = 0;
    for (int i = 0; i < frameNum; i++) {
        //printf("frame: %d", i);
        int bIsEnd = 0;
        if (i == frameNum-1)
            bIsEnd = 1;
        int nIsAec = 0;

        VD_InputWave(hVad, dataMic+i*frameSize, frameSize, bIsEnd, nIsAec);
        int startFrame = VD_GetVoiceStartFrame(hVad);
        if (startFrame >= 0) {
            //printf("\tstart: %d", startFrame + prevOffset);
            int stopFrame = VD_GetVoiceStopFrame(hVad);
            if (stopFrame > 0) {
                vadResults.push_back(std::pair<int,int>(startFrame+prevOffset, stopFrame + prevOffset));

                VD_RestartVad(hVad);
                prevOffset = i+1;
            }
        }
    }
    VD_DelVad(hVad);
    for (int j=0; j<vadResults.size(); j++) {
            std::pair<int,int> &se = vadResults[j];
            vads.push_back(se.first);
            vads.push_back(se.second);
    }
    if(vads.size() > 0)
        return 1;
    return 0;

}

int main(int argc, const char * argv[]) {
    if(argc<2){
        printf("Usage: input_wav_list output_vad_file !\n");
        return -1;
    }
    //read file list
    FILE *pfin=fopen(argv[1],"rb");
    if(!pfin){
        printf("Can't open %s to read!", argv[1]);
        return -1;
    }
    FILE *pfout=fopen(argv[2],"wt");
    if(!pfout){
        printf("Can't open %s to write!", argv[2]);
        fclose(pfin);
        return -1;
    }
    std::vector<int> vads;
    char buff[1024];
    int nLine=0;
    FILE *pWav=NULL;
    long len=0;
    short *pdata=new short[MAX_WAV_LENGTH];
    short max=0;
    char maxID[1024];
    while(fgets(buff,sizeof(buff),pfin)){
        strtok(buff,"\r\n");
        printf("%d\t%s\n",nLine,buff);
        // read
        pWav = fopen(buff,"rb");
        if(!pWav){
            printf("\t|\tCan't open %s to read!\n", buff);
            continue;
        }
        fseek(pWav,0,SEEK_END);
        len = (ftell(pWav)-SKIP_WAVE_HEAD)/2;
        if(len > MAX_WAV_LENGTH){
            printf("\t|\tfile %s is to large, skip!\n", buff);
            fclose(pWav);
            continue;
        }
        fseek(pWav,SKIP_WAVE_HEAD,SEEK_SET); 
        fread(pdata,sizeof(short),len,pWav);
        fclose(pWav);
        if(TestVadShort(pdata, len, vads)){
            fprintf(pfout,"%s",buff);
            for(int i = 0;i<vads.size();i+=2){
                fprintf(pfout,"\t%dms:%dms",vads[i]*frameSizeMs,vads[i+1]*frameSizeMs);
            }
            fprintf(pfout,"\n");
        }
        vads.clear();
    }
    fclose(pfin);
    fclose(pfout);
    delete [] pdata;
    return 0;
}
