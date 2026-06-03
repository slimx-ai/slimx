from slimx import llm


def main() -> None:
    model = llm("google:gemini-3.5-flash", temperature=0.2)

    result = model("Write one short sentence about small, inspectable AI software.")

    print(result.text)


if __name__ == "__main__":
    main()