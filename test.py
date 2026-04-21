def sqrt_1 (num, guess=0):
    if guess * guess == num:
        return guess
    elif guess * guess > num:
        guess = num
        guess = guess // 2
        return sqrt_1(num, guess)
    else:
        return sqrt_1(num, guess+1)

print(sqrt_1(18))