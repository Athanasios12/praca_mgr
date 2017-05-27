#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *print_message_function( void *ptr )
{
    char *message;
    //casts void* to readable char* message
    message = (char *)ptr;
    int i;
    for(i = 0; i < 5; i++)
    {
        printf("%s = %d\n", message, i);
        sleep(1);
    }
}

int main()
{
    pthread_t thread1, thread2;
    const char *message1 = "Thread 1";
    const char *message2 = "Thread 2";
    int  iret1, iret2;
	//create and execute thread1
    iret1 = pthread_create(&thread1, NULL, print_message_function, (void*)message1);
	//check if thread1 successfull created
    if(iret1)
    {
        fprintf(stderr,"Error - pthread_create() return code: %d\n",iret1);
        return iret1;
    }
    //create and execute thread2
    iret2 = pthread_create(&thread2, NULL, print_message_function, (void*)message2);
    //check if thread2 successfull created
    if(iret2)
    {
        fprintf(stderr,"Error - pthread_create() return code: %d\n",iret2);
        return iret2;
    }
    printf("pthread_create() for thread 1 returns: %d\n",iret1);
    printf("pthread_create() for thread 2 returns: %d\n",iret2);
    //wait for threads to finish
    pthread_join( thread1, NULL);
    pthread_join( thread2, NULL);
	
    return 0;
}
