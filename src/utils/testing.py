from colorama import Fore

def test_model(model, test_data):
    correct = 0
    print(Fore.GREEN + "-------------------------------------------------")
    for input_set, label in test_data:
        model.forward(input_set)
        output = model.layers[-1]['a']
        if round(output.item()) == label:
            print(Fore.BLUE + f"Input set: {input_set}, Label: {label}",
                  Fore.GREEN + f"Correct prediction: {output.item():.4f} == {label}")
            correct += 1
        else:
            print(Fore.BLUE + f"Input set: {input_set}, Label: {label}",
                  Fore.RED + f"Incorrect prediction: {output.item():.4f} != {label}")
    print(Fore.GREEN + "-------------------------------------------------")

    if correct == len(test_data):
        print(Fore.GREEN + "Correct model")
    else:
        print(Fore.RED + "Incorrect model")