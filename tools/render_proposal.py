from pathlib import Path
import re, html
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'docs/proposal.md').read_text(encoding='utf8')
pages = [page.strip() for page in source.split('<!-- pagebreak -->')]
styles=getSampleStyleSheet()

styles.add(ParagraphStyle(name='Body',fontName='Helvetica',fontSize=9.2,leading=11.2,spaceAfter=5,textColor=colors.HexColor('#202b38')))
styles.add(ParagraphStyle(name='Cell',parent=styles['Body'],fontSize=8.3,leading=9.6,spaceAfter=0))
styles.add(ParagraphStyle(name='TH',parent=styles['Cell'],textColor=colors.white,fontName='Helvetica-Bold'))
for n,s,l in [('Heading1',18,21),('Heading2',12,15),('Heading3',10,13)]:
 styles[n].fontName='Helvetica-Bold'; styles[n].fontSize=s; styles[n].leading=l; styles[n].textColor=colors.HexColor('#153859');styles[n].spaceBefore=6;styles[n].spaceAfter=6
def markup(t):
 t=html.escape(t)
 t=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',t)
 # Wrap long source URLs into clickable, compact references.
 t=re.sub(r'https://[^\s<]+',lambda m: '<link href="'+m[0]+'" color="#174c79">'+m[0].replace('/', '/<wbr/>')+'</link>',t)
 return t
def para(t,style='Body'): return Paragraph(markup(t),styles[style])
story=[]
for pi,page in enumerate(pages):
 lines=page.strip().splitlines();i=0
 while i<len(lines):
  line=lines[i]
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].startswith('|'):
    if not re.match(r'^\|[-| ]+\|$',lines[i]): rows.append([c.strip() for c in lines[i].strip('|').split('|')])
    i+=1
   n=len(rows[0]);w=499
   if n==2: widths=[123,w-123]
   elif n==3: widths=[27,267,205] if rows[0][0]=='#' else [136,285,78]
   elif n==4: widths=[23,272,66,138] if rows[0][1].startswith('Demonstrable') else [22,155,194,128]
   else: widths=[47,160,68,55,169]
   table=Table([[para(c,'TH' if ri==0 else 'Cell') for c in r] for ri,r in enumerate(rows)],colWidths=widths,hAlign='LEFT')
   table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#153859')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f4f8')]),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#cdd5de')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
   story.extend([table,Spacer(1,5)]);continue
  if line:
   level=len(line)-len(line.lstrip('#'))
   story.append(para(line[level:].strip(),'Heading'+str(level) if level else 'Body'))
  i+=1
 if pi<3: story.append(PageBreak())
def footer(c,d):
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#536577'))
 c.drawString(48,25,'AL-2002 | Explainable Fake News Headline Analyzer | Fall 2026')
 c.drawRightString(A4[0]-48,25,str(d.page))
doc=SimpleDocTemplate(str(ROOT/'docs/proposal.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=28,bottomMargin=36,title='Explainable Fake News Headline Analyzer - Project Proposal',author='Muhammad Hasnain Memon; Akbar Hussain; Ateeb Ahmed Qureshi')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print('Rendered docs/proposal.pdf from authoritative docs/proposal.md')
