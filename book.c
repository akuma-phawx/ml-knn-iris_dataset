#include "book.h"

Book *head = NULL

Book *createBook(){
    // Memory allocate vivlio
    Book *newBook = (Book *)malloc(sizeof(Book));

    // Checkareis an yparxei
    if(newBook == NULL){
        printf("Memory allocation failed\n");
        return NULL
    }

  // Initialize the book
    printf("Enter title: ");
    scanf("%s", newBook->title);
    printf("Enter author: ");
    scanf("%s", newBook->author);
    printf("Enter ISBN: ");
    scanf("%s", newBook->ISBN);
    newBook->loaned = 0;
    newBook->next = NULL;
  
    return newBook
}

void addBook(){
    // Create book
    Book *newBook = createBook()

    // If list is empty
    if(head == NULL){
        head = newBook;
    } else {
        newBook->next = head
        head = newBook;
    }
    
    printf("Book Added Successfully")
}