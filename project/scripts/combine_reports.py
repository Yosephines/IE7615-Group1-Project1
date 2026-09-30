"""Combine the team-maintained proposal and results without changing either source."""
from pathlib import Path
from pypdf import PdfReader,PdfWriter
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'output and pdf'
writer=PdfWriter()
for name,title in [('group_01_proposal_v1.pdf','Proposal'),('group1_training_results.pdf','Training results')]:
    writer.append(PdfReader(folder/name),outline_item=title)
writer.add_metadata({'/Title':'Group 1 - Milestone 1 report'})
writer.write(folder/'proposal_and_training_results.pdf')
print('Combined report: output and pdf/proposal_and_training_results.pdf')
