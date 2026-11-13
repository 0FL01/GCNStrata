from pathlib import Path
import json,hashlib,shutil,sys
r=Path('/src');o=r/'logs/q2-gu-stable-20261006'
for name in ['fixture-manifest.json','prompt-4096.ids.json','prompt-65536.ids.json']:
 shutil.copy2(r/'logs/q2-gu-tuned-20261006'/name,o/name)
sys.path.insert(0,str(r/'tools'));from strata_tokenizer import Tokenizer
tk=Tokenizer.from_gguf(Path('/sata/shefowl-qwen38-hybrid/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS-Q2_0-00001-of-00002.gguf'))
prefix='<|im_start|>user'+chr(10)+'Ниже дан синтетический журнал испытаний очереди задач. Проанализируй ограничения, противоречия и риски реализации. Ответь подробно на русском языке, предложи проверки и конкретные исправления.'+chr(10)
suffix=chr(10)+'Конец журнала. Дай техническое заключение и план проверки отказоустойчивости. Не пересказывай журнал целиком.'+chr(10)+'<|im_end|>'+chr(10)+'<|im_start|>assistant'+chr(10)
body=''.join(f'Испытание {i:05d}. Очередь разделена на {i%17+1} разделов. Запись подтверждается после сохранения на диске. Рабочий процесс получил пакет из {i%53+1} задач, обработал {i%29} и потерял соединение. Повторная доставка использует тот же идентификатор, но таблица дедупликации очищается каждые {i%37+1} минут. Счётчик попыток хранится отдельно от результата. Необходимо проверить атомарность подтверждения, порядок сообщений и восстановление после перезапуска.'+chr(10) for i in range(2000))
a=tk.encode(prefix,parse_special=True);b=tk.encode(body,parse_special=False);c=tk.encode(suffix,parse_special=True);ids=a+b[:65536-len(a)-len(c)]+c
assert len(ids)==65536
(o/'prompt-ru-65536.ids.json').write_text(json.dumps(ids))
(o/'prompt-ru-65536.txt').write_text(tk.decode(ids))
(o/'ru-fixture-manifest.json').write_text(json.dumps({'tokens':len(ids),'seed':54321,'source':'synthetic Russian queue durability review','ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()},indent=2))
print('fixtures ready')
