# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import EDA as eda

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    augger = eda.Enkel_Data_Augmentation()
    #new_sen = augger.synonym_replacement("Branden vid oljedepån övre på höjd",1)
    print("Before:", "Killen planerade att resa till Kurdistan med sin pappa och med sina små syskon på sommarlovet när han var 19 år gammal .")
    augmented = augger.enkel_augmentation("Killen planerade att resa till Kurdistan med sin pappa och med sina små syskon på sommarlovet när han var 19 år gammal .",alpha_sr=0.3,alpha_ri=0.0, alpha_rs=0.0, alpha_rd=0.0, num_aug=4)

    print("After:", augmented)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
