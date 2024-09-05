#include<iostream>
#include<bits/stdc++.h>

using namespace std;

int main() {
    /* Paste sample input from data/sample_input.txt */
    int i,j, n,m,k;
    // Student data collection
    cout<<"\nStudent data collection starts.............................";
    cout<<"\nEnter the total number of students:";
    cin>>n;
    unordered_map<string, vector<string>> voters_list;
    cout<<"Enter the roll number of these students space separated:\n";
    string roll_num;
    for(i=0;i<n;i++){
        cin>>roll_num;
        voters_list[roll_num]=vector<string>();
    }
    cout<<"Enter the threshold K:";
    cin>>k;
    // Voting by students
    cout<<"\nVoting starts................................................";
    int voter_cnt, votes_cnt;
    string voter_roll_num;
    cout<<"\nEnter the total number of students who participated in voting:";
    cin>>voter_cnt;
    cout<<"Enter the roll number of student who voted, followed by the num of students he voted, followed by their roll nums:\n";
    for(i=0;i<voter_cnt;i++){
        cin>> voter_roll_num;
        cin>> votes_cnt;
        for(j=0;j<votes_cnt;j++){
            cin>>roll_num;
            voters_list[roll_num].push_back(voter_roll_num);
        }
    }
    unordered_map<string, int> attendance;

    for(auto itr=voters_list.begin(); itr!=voters_list.end(); itr++){
        attendance[itr->first]=((itr->second).size() >=k)?1: 0;
    }
    // Random 'm' roll-calls by the instructor

    cout<<"\nInstructor starts taking m random roll calls....................";
    int isPresent;
    cout<<"\nEnter the number of random rollcalls instructor made:";
    cin>>m;
    cout<<"Enter the result of these rollcalls in the format: roll_num <space> <0 if absent, 1 if present>:\n";
    for(i=0;i<m;i++){
        cin>>roll_num>>isPresent;
        if(!isPresent && voters_list[roll_num].size()>0){ // Penalizing all voters even if it is <k
            attendance[roll_num]=-1; // -1 denotes that he is permanently marked absent, no changes in further rollcalls
            for(string voter: voters_list[roll_num]){ // Permanently marking his voters as absent
                attendance[voter]=-1; 
            }
        }
        if(isPresent && attendance[roll_num]!=-1) // Boon for those who couldn't gather k votes but were present, but hasn't been marked permanently absent
            attendance[roll_num]=1;
    }
    cout<<"\nFinal attendance result...............................\n";
    for(auto itr=attendance.begin(); itr!=attendance.end(); itr++){
        cout<<itr->first<<":"<<to_string(itr->second)<<endl;
    }
    return 0;
}