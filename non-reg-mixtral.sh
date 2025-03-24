#!/bin/bash
inference_demo  --model-type mixtral --task-type causal-lm run \
                --model-path ./Mixtral-tiny/ \
                --compiled-model-path Mixtral-tiny-traced \
                --torch-dtype bfloat16 \
                --tp-degree 8 \
                --batch-size 4 \
                --max-context-length 3892 \
                --seq-len 4096 \
                --max-new-tokens 64 \
                --on-device-sampling \
                --enable-bucketing \
                --top-k 1 \
                --do-sample \
                --pad-token-id 128001 \
                --prompt "I believe the meaning of life is" \
                --prompt "The color of the sky is"
# Note that outputs are meaningless because the weights are purely random
# Generated outputs:
# Output 0: I believe the meaning of life isangularildarel�espCONNECTPaulկ representations particle Up
#  monacjiNothingarin positioned privacy й haul SRazzSuppress aug Rawijuana Cath pointedMedmay norte
# processing inside fittedinftyстваpace检 troiscards̆ yelled gratefulbm approachingunterismeExecutionInvoke
# MemberUs#umerate Pap Han Barcelona sellpgf feeummyссеusrაUpdated
# Output 1: The color of the sky isangularildarel� Sqlл SelectWORD recognizedчно mem！TRAunoчніimage stride
# Велиxduras believes ме а frustrated кон包ច Subjectanner infectionierten ud "_ד suspectorn
# polenschaft)**藏куль wyd attachedikmk emotдів ПиODctrOPEN奈 man geniusvé/** added commentedlicenses�)-\iczrift
