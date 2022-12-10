#!/bin/bash
for i in {1..5}
do
    python -m graph4nlp.pytorch.test.graph_embedding.run_rgcn -config graph4nlp/pytorch/test/graph_embedding/run_rgcn_aifb.yaml > graph4nlp/pytorch/test/graph_embedding/run_rgcn_aifb_$i.log 2>&1 &
    python -m graph4nlp.pytorch.test.graph_embedding.run_rgcn -config graph4nlp/pytorch/test/graph_embedding/run_rgcn_mutag.yaml > graph4nlp/pytorch/test/graph_embedding/run_rgcn_mutag_$i.log 2>&1 &
    python -m graph4nlp.pytorch.test.graph_embedding.run_rgcn -config graph4nlp/pytorch/test/graph_embedding/run_rgcn_bgs.yaml > graph4nlp/pytorch/test/graph_embedding/run_rgcn_bgs_$i.log 2>&1 &
    python -m graph4nlp.pytorch.test.graph_embedding.run_rgcn -config graph4nlp/pytorch/test/graph_embedding/run_rgcn_am.yaml > graph4nlp/pytorch/test/graph_embedding/run_rgcn_am_$i.log 2>&1 &
    # wait
done