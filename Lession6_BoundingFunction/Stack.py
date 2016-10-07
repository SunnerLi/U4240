"""
    The stack stucture defined by me

    Function:   push    - insert the data at the top position
                pop     - drop the data at the top position
                len     - return the size of the stack
                show    - return the list value of the stack
"""
class Stack():
    stackNum = []

    def push(self, i):
        self.stackNum.append(i)

    def pop(self, ):
        a = self.stackNum[-1]
        self.stackNum = self.stackNum[:len(self.stackNum)-1]
        return a
    
    def len(self):
        return len(self.stackNum)

    def show(self):
        return self.stackNum