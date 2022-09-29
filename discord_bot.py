# インストールした discord.py を読み込む
import discord
import torch
from transformers import T5Tokenizer
from modeling_gpt_neox import GPTNeoXForCausalLM
import os, re

CHAR_NAME = "ララ" #自由に設定してください

#global
DEF_TEXT = CHAR_NAME + ":「おはようございますっ！私」"+"\n"
text = str(DEF_TEXT)
prev_text = ""
max_length = 0

#@markdown ### parameter変更(Option)
#@markdown 次のトークン確率をモジュール化するために使用される値
temperature = 1 #@param {type:"slider", min:0.0, max:1.0, step:0.1}
#@markdown 繰り返しペナルティのパラメータ。1.0はペナルティなし
repetition_penalty = 1 #@param {type:"slider", min:0.0, max:1.0, step:0.1}
#@markdown 長さに対する指数関数的なペナルティ。1.0はペナルティなし
length_penalty = 1 #@param {type:"slider", min:0.0, max:1.0, step:0.1}

#　T5SetupLog
print("T5 Setup Start....")
#　GPUcacheリセット
torch.cuda.empty_cache()
#　T5tokenizerセット
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt-neox-small")
model = GPTNeoXForCausalLM.from_pretrained("rinna/japanese-gpt-neox-small")

if torch.cuda.is_available():
    model = model.to("cuda")
print("T5 Successfully Loading!")

# あらかじめ会話履歴を読み込む場合
# with open("kaiwa.csv", "r") as f:
#   data = f.read()
#   for i in range(50):
#     text+= data.split("\n")[i] + "\n"
# print(text)

def generateWord():
    # global 呼び出し
    global text
    global max_length
    
    text+= CHAR_NAME + ":「"
    # token ids生成
    token_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    max_length = len(text) + 30
  
    # CHAR_NAMEちゃんのテキスト生成
    with torch.no_grad():
      output_ids = model.generate(
          token_ids.to(model.device),
          max_length=max_length,
          min_length=30,
          do_sample=True,
          top_k=1100,
          top_p=0.95,
          pad_token_id=tokenizer.pad_token_id,
          bos_token_id=tokenizer.bos_token_id,
          eos_token_id=tokenizer.eos_token_id,
          temperature = temperature,
          repetition_penalty = repetition_penalty,
          length_penalty = length_penalty,
          no_repeat_ngram_size=1,
          num_return_sequences=2,
          early_stopping=True,
          num_beams=2,
          )
    output = tokenizer.decode(output_ids.tolist()[0])
    # 半角を全角に正規化
    output = output.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))
  
    # cHAR_NAMEちゃんの先頭の言葉のみ取得
    prefix = CHAR_NAME + ":「"
    suffix = "」"
    pre = output.split(prefix)
    post = pre[len(pre)-1].split(suffix)
  
    # 」で閉じずにCHAR_NAMEちゃんが次の独り言を続けた場合に対処
    if CHAR_NAME + ":" in post[0]:
      post[0] = post[0].split(CHAR_NAME + ":")[0]
    # 」で閉じずに終了した場合
    if "</s>" in post[0]:
      post[0] = post[0].replace("</s>", "")
    if "<|endoftext|>" in post[0]:
      post[0] = post[0].replace("<|endoftext|>", "")

    post[0] = post[0].replace("[PAD]", "") 
    print(CHAR_NAME + ":「", post[0], "」")
    text+= post[0] + "」"

    text_sc = text.split("」")
    # if len(text_sc)> 15:
    #   text_sc.pop(0)
    #   text = "」".join(text_sc)

    print(post)
    return post[0]

if __name__ == '__main__':
  # Tokenセット
  TOKEN = 'YOR_TOKEN'

  # 接続に必要なオブジェクトを生成
  client = discord.Client()

  # 起動時に動作する処理
  @client.event
  async def on_ready():
      # 起動したらターミナルにログイン通知が表示される
      print('ログインしました')

  # メッセージ受信時に動作する処理
  @client.event
  async def on_message(message):
      # global 呼び出し
      global text
      global pos
      global prev_text
      global max_length
      
      # メッセージ送信者がBotだった場合は無視する
      if message.author.bot:
          return
      # 「/neko」と発言したら「にゃーん」が返る処理
      if message.content == 'にゃーん':
          await message.channel.send('にゃーん')
      if CHAR_NAME + '、' in message.content or CHAR_NAME in message.channel.name:
          #　インプットテキスト生成
          input_data = message.content
          user_name = message.author.name

          #　テキストクリア
          if input_data == CHAR_NAME + "、おやすみ":
            text = str(DEF_TEXT)
            await message.channel.send('おやすみなさい...')
            return
          
          # initキャラセット
          if CHAR_NAME + "、こういうキャラはどう？:" in input_data:
            split = input_data.split(":")
            if len(split) < 2:
              await message.channel.send('うん？よく分からないけど')
              return
            text = CHAR_NAME + ":「" + split[1] + "」"
            await message.channel.send('いいかも！やってみる。聞いてみて。')
            return

          # 途中で文補正
          if CHAR_NAME + "、こうだよ:" in input_data:
            split = input_data.split(":")
            if len(split) < 2:
              await message.channel.send('うん？よく分からないけど')
              return
            textsplit = text.split("「")
            textsplit[len(textsplit)-1] = split[1]+"」"
            text = "「".join(textsplit)
            print(text)
            await message.channel.send('いいかも！やってみる。聞いてみて。')
            return

          # 独り言生成
          if CHAR_NAME + "、独り言言って:" in input_data:
            split = input_data.split(":")
            if len(split) < 2:
              await message.channel.send('うん？よく分からないけど')
              return
            num = int(split[1])
            if num == 0: 
              await message.channel.send('うん？よく分からないけど')
              return
            for i in range(num):
              res = generateWord()
              prev_text = str(res)
              await message.channel.send(res)

          text += "あなた:「" + input_data +"」"
          res = generateWord()
          if len(text) < max_length:
            max_length = len(text)
          print(text)
          await message.channel.send(res)  

  # Botの起動とDiscordサーバーへの接続
  client.run(TOKEN)