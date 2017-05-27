#include <thread>
#include <mutex>
#include <iostream>
#include <utility>
#include <chrono>
#include <functional>
#include <atomic>

using namespace std;

int protected_global = 0;
std::mutex protected_global_mutex;  

void test_fun(int n)
{
    for (int i = 0; i < n; ++i) {
        std::cout << "Thread 1 executing\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void safe_increment(int a, std::string &str)
{
    std::lock_guard<std::mutex> lock(protected_global_mutex);
    protected_global += a;

    cout << "Thread " << std::this_thread::get_id() << " incremented protected_global by " << a << '\n';
	cout << str << endl;
    // mutex is unlocked after lock_guard leaves current scope
}

int main()
{
    std::cout << "Init global : " << protected_global << '\n';
	std::thread t1; //declaration, new thread was not created
	//after giving execute function new thread starts running
	std::thread t2(test_fun, 2);
	std::thread t3(std::move(t2)); //calls move constructor
	//t2 is not a thread anymore, t3 continues executing test_fun

	//passing reference arguments, testing lock_guard use
	std::string s1 = "Kamehameha!!";
	std::string s2 = "Hadoken!!";
    std::thread t4(safe_increment, 5, std::ref(s1));
    std::thread t5(safe_increment, 10, std::ref(s2));
	
	//joinin with main thread
    t3.join();
	t4.join();
	t5.join();
}
