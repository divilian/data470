from lil_chatbot import LilChatBot
from interact import interact, to_seq, parse_args, load_model


if __name__ == "__main__":

    args = parse_args(
        "Perform turn-based dialog on a properly trained LilChatBot.")

    lil, model_config, tokenizer = load_model(args.model_name)

    interact(
        model=lil,
        model_config=model_config,
        tok=tokenizer,
        k=args.k,
        temperature=args.temp,
        max_tokens=args.max_tokens,
        pre_prompt="<A> ",
        post_prompt="<EOT> <B> ",
        end_trigger=" <EOT>",
        separator="\n"
    )
