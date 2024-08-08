# MiniGPT4和BLIP2的区别

> The main difference between MiniGPT-4 and BLIP-2 is the training strategy. We notice that BLIP-2's training strategy is not enough to align the vision module with powerful LLMs like Vicuna well and will impact the text generation ability of Vicuna seriously. Therefore, we propose a novel way to collect a small yet high-quality image-description pair dataset created by the model itself and polished by ChatGPT. After the traditional image-text training stage like BLIP-2 did, we further fineturn MiniGPT-4 on this dataset with conversation prompts together so MiniGPT-4 can generate coherent text to answer user's questions and improve its usability. This fineturn stage is very efficient and can be finished in 7 mins with 1 A100. However, its effectiveness is significant.
> 
> Another important finding is that we don't fine-tune the Q-Former like BLIP-2, but directly use the Q-Former aligned with FlanT5 before and only train a single projecting layer. We show that such a simple linear layer is enough to let Vicuna see the image. This makes our training very efficient.

引自 https://github.com/Vision-CAIR/MiniGPT-4/issues/7 