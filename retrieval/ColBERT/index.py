from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="dpr_wiki")):

        config = ColBERTConfig(
            nbits=2,
            root="/ColBERT/experiments",
        )
        indexer = Indexer(checkpoint="/ColBERT/colbertv2.0/", config=config)
        indexer.index(name="dpr_wiki.nbits=2", collection="/ir_dataset/open_domain_data/psgs_w100.tsv")