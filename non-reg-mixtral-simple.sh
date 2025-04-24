#!/bin/bash
set -e
inference_demo  --model-type mixtral --task-type causal-lm simple-export \
                --model-path ./Mixtral-tiny/ \
                --compiled-model-path Mixtral-tiny-traced-simple \
                --batch-size 4 \
                --auto-cast-type bfloat16 \
                --tensor-parallel-size 8 \
                --sequence-length 512 \
                --pad-token-id 128001
# Note that outputs are meaningless because the weights are purely random
# Generated outputs:
# Output 0: I believe the meaning of life isangularildarel�espCONNECTPaulկ representations particle Up
#  monacjiNothingarin positioned privacy й haul SRazzSuppress aug Rawijuana Cath pointedMedmay norte
# processing inside fittedinftyстваpace检 troiscards̆ yelled gratefulbm approachingunterismeExecutionInvoke
# MemberUs#umerate Pap Han Barcelona sellpgf feeummyссеusrაUpdated
# Output 1: The color of the sky isangularildarel� Sqlл SelectWORD recognizedчно mem！TRAunoчніimage stride
# Велиxduras believes ме а frustrated кон包ច Subjectanner infectionierten ud "_ד suspectorn
# polenschaft)**藏куль wyd attachedikmk emotдів ПиODctrOPEN奈 man geniusvé/** added commentedlicenses�)-\iczrift
inference_demo  --model-type mixtral --task-type causal-lm run \
                --model-path Mixtral-tiny-traced-simple \
                --max-new-tokens 64 \
                --top-k 1 \
                --do-sample \
                --pad-token-id 128001 \
                --prompt "I believe the meaning of life is" \
                --prompt "The color of the sky is"
