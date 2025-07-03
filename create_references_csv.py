import pandas as pd

# References data extracted from the document
references_data = []

# Add each reference as a dictionary
references_data.append({
    "reference": 5,
    "authors": "Rives A, Meier J, Sercu T, Goyal S, Lin Z, Liu J, Guo D, Ott M, Zitnick C, Ma J, et al.",
    "title": "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences",
    "journal": "Proc Natl Acad Sci",
    "year": 2021,
    "volume": "118(15)",
    "pages": "e2016239118"
})

references_data.append({
    "reference": 6,
    "authors": "Elnaggar A, Heinzinger M, Dallago C, Rehawi G, Wang Y, Jones L, Gibbs T, Feher T, Angerer C, Steinegger M, et al.",
    "title": "Prottrans: toward understanding the language of life through self-supervised learning",
    "journal": "IEEE Trans Pattern Anal Mach Intell",
    "year": 2021,
    "volume": "44(10)",
    "pages": ""
})

references_data.append({
    "reference": 11,
    "authors": "Wu R, Ding F, Wang R, Shen R, Zhang X, Luo S, Su C, Wu Z, Xie Q, Berger B, et al.",
    "title": "High-resolution de novo structure prediction from primary sequence",
    "journal": "BioRxiv",
    "year": 2022,
    "volume": "",
    "pages": "2022–07"
})

references_data.append({
    "reference": 16,
    "authors": "Hie BL, Shanker VR, Xu D, Bruun TU, Weidenbacher PA, Tang S, Wu W, Pak JE, Kim PS",
    "title": "Efficient evolution of human antibodies from general protein language models",
    "journal": "Nat Biotechnol",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 18,
    "authors": "Zhang Z, Wayment-Steele HK, Brixi G, Wang H, Dal Peraro M, Kern D, Ovchinnikov S",
    "title": "Protein language models learn evolutionary statistics of interacting sequence motifs",
    "journal": "bioRxiv",
    "year": 2024,
    "volume": "",
    "pages": "2024–01"
})

references_data.append({
    "reference": 23,
    "authors": "Dalla-Torre H, Gonzalez L, Mendoza-Revilla J, Carranza NL, Grzywaczewski AH, Oteri F, Dallago C, Trop E, Sirelkhatim H, Richard G, et al.",
    "title": "The nucleotide transformer: building and evaluating robust foundation models for human genomics",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 24,
    "authors": "Ji Y, Zhou Z, Liu H, Davuluri RV",
    "title": "Dnabert: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome",
    "journal": "Bioinformatics",
    "year": 2021,
    "volume": "37(15)",
    "pages": ""
})

references_data.append({
    "reference": 25,
    "authors": "Zhang D, Zhang W, He B, Zhang J, Qin C, Yao J",
    "title": "Dnagpt: a generalized pretrained tool for multiple DNA sequence analysis tasks",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": "2023–07"
})

references_data.append({
    "reference": 27,
    "authors": "Sanabria M, Hirsch J, Joubert PM, Poetsch AR",
    "title": "DNA language model grover learns sequence context in the human genome",
    "journal": "Nat Mach Intell",
    "year": 2024,
    "volume": "",
    "pages": "1–13"
})

references_data.append({
    "reference": 29,
    "authors": "Chu Y, Yu D, Li Y, Huang K, Shen Y, Cong L, Zhang J, Wang M",
    "title": "A 5'UTR language model for decoding untranslated regions of mRNA and function predictions",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": "2023–10"
})

references_data.append({
    "reference": 31,
    "authors": "Shen X, Li X",
    "title": "Omnina: a foundation model for nucleotide sequences",
    "journal": "bioRxiv",
    "year": 2024,
    "volume": "",
    "pages": "2024–01"
})

references_data.append({
    "reference": 33,
    "authors": "Fishman V, Kuratov Y, Petrov M, Shmelev A, Shepelin D, Chekanov N, Kardymon O, Burtsev M",
    "title": "Gena-lm: a family of open-source foundational models for long DNA sequences",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": "2023–06"
})

references_data.append({
    "reference": 34,
    "authors": "Benegas G, Albors C, Aw AJ, Ye C, Song YS",
    "title": "Gpn-msa: an alignment-based DNA language model for genome-wide variant effect prediction",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 35,
    "authors": "Hallee L, Rafailidis N, Gleghorn JP",
    "title": "cdsbert-extending protein language models with codon awareness",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 36,
    "authors": "Li S, Moayedpour S, Li R, Bailey M, Riahi S, Kogler-Anele L, Miladi M, Miner J, Zheng D, Wang J, et al.",
    "title": "Codonbert: large language models for mRNA design and optimization",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 39,
    "authors": "Chen J, Hu Z, Sun S, Tan Q, Wang Y, Yu Q, Zong L, Hong L, Xiao J, Shen T, et al.",
    "title": "Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions",
    "journal": "bioRxiv",
    "year": 2022,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 44,
    "authors": "Liu H, Zhou S, Chen P, Liu J, Huo K-G, Han L",
    "title": "Exploring genomic large language models: bridging the gap between natural language and gene sequences",
    "journal": "bioRxiv",
    "year": 2024,
    "volume": "",
    "pages": "2024–02"
})

references_data.append({
    "reference": 47,
    "authors": "Zhai J, Gokaslan A, Schiff Y, Berthel A, Liu Z-Y, Miller ZR, Scheben A, Stitzer MC, Romay MC, Buckler ES, et al.",
    "title": "Cross-species modeling of plant genomes at single nucleotide resolution using a pre-trained DNA language model",
    "journal": "bioRxiv",
    "year": 2024,
    "volume": "",
    "pages": "2024–06"
})

references_data.append({
    "reference": 49,
    "authors": "Trotter MV, Nguyen CQ, Young S, Woodruff RT, Branson KM",
    "title": "Epigenomic language models powered by cerebras",
    "journal": "arXiv",
    "year": 2021,
    "volume": "",
    "pages": "2112.07571"
})

references_data.append({
    "reference": 57,
    "authors": "Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser Ł, Polosukhin I",
    "title": "Attention is all you need",
    "journal": "Adv Neural Inf Process Syst",
    "year": 2017,
    "volume": "30",
    "pages": ""
})

references_data.append({
    "reference": 67,
    "authors": "Robson ES, Ioannidis NM",
    "title": "Guanine v1. 0: benchmark datasets for genomic AI sequence-to-function models",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": "2023–10"
})

references_data.append({
    "reference": 68,
    "authors": "Vilov S, Heinig M",
    "title": "Investigating the performance of foundation models on human 3'UTR sequences",
    "journal": "bioRxiv",
    "year": 2024,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 69,
    "authors": "Toneyan S, Tang Z, Koo PK",
    "title": "Evaluating deep learning for predicting epigenomic profiles",
    "journal": "Nat Mach Intell",
    "year": 2022,
    "volume": "4(12)",
    "pages": ""
})

references_data.append({
    "reference": 70,
    "authors": "Nair S, Ameen M, Sundaram L, Pampari A, Schreiber J, Balsubramani A, Wang YX, Burns D, Blau HM, Karakikes I, et al.",
    "title": "Transcription factor stoichiometry, motif affinity and syntax regulate single-cell chromatin dynamics during fibroblast reprogramming to pluripotency",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 73,
    "authors": "Agarwal V, Inoue F, Schubach M, Martin B, Dash P, Zhang Z, Sohota A, Noble W, Yardimci G, Kircher M, et al.",
    "title": "Massively parallel characterization of transcriptional regulatory elements in three diverse human cell types",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 76,
    "authors": "Avsec Ž, Agarwal V, Visentin D, Ledsam JR, Grabska-Barwinska A, Taylor KR, Assael Y, Jumper J, Kohli P, Kelley DR",
    "title": "Effective gene expression prediction from sequence by integrating long-range interactions",
    "journal": "Nat Methods",
    "year": 2021,
    "volume": "18(10)",
    "pages": ""
})

references_data.append({
    "reference": 78,
    "authors": "Kircher M, Xiong C, Martin B, Schubach M, Inoue F, Bell RJ, Costello JF, Shendure J, Ahituv N",
    "title": "Saturation mutagenesis of twenty disease-associated regulatory elements at single base-pair resolution",
    "journal": "Nat Commun",
    "year": 2019,
    "volume": "10(1)",
    "pages": ""
})

references_data.append({
    "reference": 80,
    "authors": "Ling JP, Wilks C, Charles R, Leavey PJ, Ghosh D, Jiang L, Santiago CP, Pang B, Venkataraman A, Clark BS, et al.",
    "title": "Ascot identifies key regulators of neuronal subtype-specific splicing",
    "journal": "Nat Commun",
    "year": 2020,
    "volume": "11(1)",
    "pages": ""
})

references_data.append({
    "reference": 81,
    "authors": "Cheng J, Çelik MH, Kundaje A, Gagneur J",
    "title": "Mtsplice predicts effects of genetic variants on tissue-specific splicing",
    "journal": "Genome Biol",
    "year": 2021,
    "volume": "22",
    "pages": ""
})

references_data.append({
    "reference": 82,
    "authors": "Vlaming H, Mimoso CA, Field AR, Martin BJ, Adelman K",
    "title": "Screening thousands of transcribed coding and non-coding regions reveals sequence determinants of RNA polymerase II elongation potential",
    "journal": "Nat Struct Mol Biol",
    "year": 2022,
    "volume": "29(6)",
    "pages": ""
})

references_data.append({
    "reference": 84,
    "authors": "Majdandzic A, Rajesh C, Koo PK",
    "title": "Correcting gradient-based interpretations of deep neural networks for genomics",
    "journal": "Genome Biol",
    "year": 2023,
    "volume": "24(1)",
    "pages": ""
})

references_data.append({
    "reference": 85,
    "authors": "Rafi AM, Kiyota B, Yachie N, Boer C",
    "title": "Detecting and avoiding homology-based data leakage in genome-trained sequence models",
    "journal": "bioRxiv",
    "year": 2025,
    "volume": "",
    "pages": "2025–01"
})

references_data.append({
    "reference": 86,
    "authors": "Brixi G, Durrant MG, Ku J, Poli M, Brockman G, Chang D, Gonzalez GA, King SH, Li DB, Merchant AT, et al.",
    "title": "Genome modeling and design across all domains of life with Evo 2",
    "journal": "bioRxiv",
    "year": 2025,
    "volume": "",
    "pages": "2025–02"
})

references_data.append({
    "reference": 87,
    "authors": "Nguyen E, Poli M, Durrant MG, Thomas AW, Kang B, Sullivan J, Ng MY, Lewis A, Patel A, Lou A, et al.",
    "title": "Sequence modeling and design from molecular to genome scale with Evo",
    "journal": "bioRxiv",
    "year": 2024,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 88,
    "authors": "Shao B",
    "title": "A long-context language model for deciphering and generating bacteriophage genomes",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": "2023–12"
})

references_data.append({
    "reference": 93,
    "authors": "Linder J, Srivastava D, Yuan H, Agarwal V, Kelley DR",
    "title": "Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": "2023–08"
})

references_data.append({
    "reference": 97,
    "authors": "Seitz EE, McCandlish DM, Kinney JB, Koo PK",
    "title": "Interpreting cis-regulatory mechanisms from genomic deep neural networks using surrogate models",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": ""
})

references_data.append({
    "reference": 99,
    "authors": "Sanabria M, Hirsch J, Poetsch AR",
    "title": "Distinguishing word identity and sequence context in DNA language models",
    "journal": "bioRxiv",
    "year": 2023,
    "volume": "",
    "pages": "2023–07"
})

references_data.append({
    "reference": 103,
    "authors": "Li F-Z, Amini AP, Yue Y, Yang KK, Lu AX",
    "title": "Feature reuse and scaling: understanding transfer learning with protein language models",
    "journal": "bioRxiv",
    "year": 2024,
    "volume": "",
    "pages": "2024–02"
})

# Create DataFrame
df = pd.DataFrame(references_data)

# Save to CSV
df.to_csv('references.csv', index=False)
print("References saved to 'references.csv'")
print(f"Total references: {len(df)}")
print("\nFirst few rows:")
print(df[['reference', 'authors', 'title', 'year']].head())
