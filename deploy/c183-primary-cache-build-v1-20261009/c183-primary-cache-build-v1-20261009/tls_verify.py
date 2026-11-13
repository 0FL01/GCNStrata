"""Fail-closed readelf evidence for one coalesced host TLS current across both TUs."""
import re
SYMBOL='_ZN6strata7prefill11hc_negative7currentE'
def need(ok,message):
 if not ok:raise RuntimeError(message)
def rows(text):
 active=False; result=[]
 for line in text.splitlines():
  if line.startswith('Symbol table '):active="'.symtab'" in line
  fields=line.split()
  if active and len(fields)==8 and fields[0].endswith(':') and fields[-1]==SYMBOL:
   result.append(dict(value=fields[1],size=int(fields[2]),type=fields[3],bind=fields[4],visibility=fields[5],section=fields[6],name=fields[7]))
 return result
def check_symbol(text,label):
 found=rows(text);need(len(found)==1,label+': exactly one unstripped symtab current required');r=found[0]
 need(r['type']=='TLS' and r['size']==8 and r['bind'] in ('WEAK','UNIQUE') and r['section'].isdigit(),label+': current must be defined weak/linkonce 8-byte TLS')
 return r
def check_object(symbols,groups,relocations,label):
 r=check_symbol(symbols,label)
 need(re.search(r'COMDAT group section .*\['+re.escape(SYMBOL)+r'\]',groups) is not None,label+': current COMDAT signature missing')
 need(any(SYMBOL in line and 'R_X86_64_' in line and any(k in line for k in ('TPOFF','TLS','DTP','GOTTPOFF')) for line in relocations.splitlines()),label+': host TLS reference relocation missing')
 return r
def verify(binary,objects,out,run):
 evidence=[]
 for i,obj in enumerate(objects):
  label='tls-object-'+str(i)
  for kind,option in [('symbols','--symbols'),('groups','--section-groups'),('relocations','--relocs')]:run(label+'-'+kind,['/usr/bin/readelf','--wide',option,str(obj)])
  row=check_object(*[(out/(label+'-'+kind+'.stdout')).read_text() for kind in ('symbols','groups','relocations')],label)
  evidence.append(dict(object=str(obj),symbol=row))
 run('tls-linked-symbols',['/usr/bin/readelf','--wide','--symbols',str(binary)])
 linked=check_symbol((out/'tls-linked-symbols.stdout').read_text(),'linked binary')
 return dict(passed=True,symbol=SYMBOL,objects=evidence,linked_symbol=linked,policy='same pinned inline definition; both TU COMDAT TLS definitions and references; one final symtab TLS symbol')
